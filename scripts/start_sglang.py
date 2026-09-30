#!/usr/bin/env python3
"""Start SGLang's OpenAI-compatible server using the same .env as vLLM."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from common.bootstrap import PROJECT_ROOT, ensure_sys_path

ensure_sys_path()

from common.env import ConfigError, apply_runtime_env, load_app_config
from common.health import check_openai_models
from common.logging_utils import setup_logging
from common.platform_utils import is_windows
from common.preflight import run_preflight
from common.process import (
    current_python,
    is_pid_running,
    log_file,
    pid_file,
    port_in_use,
    read_pid,
    start_process,
    wait_for_or_exit,
    write_pid,
)
from common.secrets import redact_command
from common.vllm_launcher import VLLMLaunchError, detect_trained_mode
from common.windows_vllm_runtime import default_native_health_timeout, log_tail, serving_child_env

LOGGER = setup_logging("llm_tools.sglang")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start an OpenAI-compatible SGLang server.")
    parser.add_argument("--foreground", action="store_true")
    parser.add_argument("--daemon", action="store_true")
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", default=None)
    parser.add_argument("--service-name", default="sglang")
    parser.add_argument("--trained", action="store_true", help="Load CHECKPOINT/ADAPTER like start_vllm_trained.py.")
    parser.add_argument("--mode", choices=["auto", "lora", "merged"], default=None)
    parser.add_argument("extra", nargs=argparse.REMAINDER, help="Extra args forwarded to sglang.launch_server after `--`.")
    return parser.parse_args()


def build_sglang_command(
    python_executable: str,
    model_path: Path,
    config,
    extra_args: list[str] | None = None,
    lora_path: Path | None = None,
) -> list[str]:
    host = config.require("VLLM_HOST")
    port = config.require("VLLM_PORT")
    cmd = [
        python_executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        str(model_path),
        "--host",
        host,
        "--port",
        str(port),
        "--tp",
        str(config.get("TENSOR_PARALLEL_SIZE", "1")),
        "--mem-fraction-static",
        str(config.get("GPU_MEMORY_UTILIZATION", "0.9")),
        "--context-length",
        str(config.get("MAX_MODEL_LEN", "4096")),
        "--dtype",
        str(config.get("DTYPE", "auto")),
    ]
    api_key = config.get("VLLM_API_KEY")
    if api_key:
        cmd += ["--api-key", api_key]
    served = config.get("VLLM_SERVED_MODEL_NAME")
    if served:
        cmd += ["--served-model-name", served]
    quant = (config.get("QUANTIZATION") or "").strip()
    if quant:
        cmd += ["--quantization", quant]
    if config.get_bool("VLLM_TRUST_REMOTE_CODE", True):
        cmd.append("--trust-remote-code")
    tool_parser = (config.get("SGLANG_TOOL_CALL_PARSER") or config.get("VLLM_TOOL_CALL_PARSER") or "").strip()
    if tool_parser:
        cmd += ["--tool-call-parser", tool_parser]
    reasoning = (config.get("SGLANG_REASONING_PARSER") or "").strip()
    if reasoning:
        cmd += ["--reasoning-parser", reasoning]
    if lora_path is not None:
        cmd += ["--lora-paths", str(lora_path), "--max-loras-per-batch", "1"]
        LOGGER.warning(
            "Passing LoRA to SGLang as --lora-paths. If this build rejects it, merge first: "
            "uv run python scripts/merge_lora.py"
        )
    if extra_args:
        cmd += extra_args
    return cmd


def ensure_sglang_importable() -> None:
    import subprocess

    completed = subprocess.run(
        [current_python(), "-c", "import sglang.launch_server"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if completed.returncode == 0:
        return
    tail = (completed.stderr or completed.stdout or "").strip().splitlines()
    detail = tail[-1][:400] if tail else "import failed"
    raise VLLMLaunchError(
        "SGLang is not installed in this environment. "
        "vLLM and SGLang pin different torch builds, so switch the same .venv with:\n"
        "  uv sync --directory overlays/sglang\n"
        "Switch back to vLLM with `uv sync --extra infer`. "
        f"Do not install both at once.\nImport error: {detail}"
    )


def _health_ok(host: str, port: int, api_key: str | None) -> bool:
    try:
        check_openai_models(host, port, api_key=api_key, timeout=3)
        return True
    except Exception:
        return False


def main() -> int:
    args = parse_args()
    try:
        if is_windows():
            LOGGER.warning(
                "SGLang is a Linux server. On native Windows this import usually fails; use WSL2 or Docker."
            )
        config = load_app_config()
        apply_runtime_env(config)
        config.require_keys(["VLLM_HOST", "VLLM_PORT"])
        config.override(VLLM_HOST=args.host, VLLM_PORT=str(args.port) if args.port else None)

        lora_path = None
        if args.trained:
            mode, model_path, adapter = detect_trained_mode(config, requested=args.mode)
            LOGGER.info("SGLang trained mode=%s model=%s adapter=%s", mode, model_path, adapter)
            if mode == "lora":
                lora_path = adapter
            model_dir = model_path
        else:
            config.require_keys(["MODEL_DIR"])
            model_dir = Path(args.model_dir) if args.model_dir else config.require_path("MODEL_DIR")

        if not model_dir.exists():
            raise ConfigError(f"Model directory does not exist: {model_dir}")

        report = run_preflight(
            model_dir,
            max_model_len=int(config.get("MAX_MODEL_LEN") or 4096),
            dtype=config.get("DTYPE"),
            gpu_memory_utilization=float(config.get("GPU_MEMORY_UTILIZATION") or 0.9),
        )
        for warning in report.warnings:
            LOGGER.warning("%s", warning)
        if not report.ok:
            raise ConfigError("Model failed preflight: " + "; ".join(report.problems))

        host = config.require("VLLM_HOST")
        port = int(config.require("VLLM_PORT"))
        if port_in_use(host, port):
            raise ConfigError(f"Port {port} is already in use on {host}.")

        pid_dir = config.require_path("PID_DIR")
        log_dir = config.require_path("LOG_DIR")
        service_name = args.service_name or "sglang"
        existing = read_pid(pid_file(pid_dir, service_name))
        if existing and is_pid_running(existing):
            raise ConfigError(f"{service_name} already running as PID {existing}")

        ensure_sglang_importable()
        extra = [item for item in args.extra if item != "--"]
        cmd = build_sglang_command(current_python(), model_dir, config, extra, lora_path)
        LOGGER.info("SGLang command: %s", " ".join(redact_command(cmd)))

        daemon = bool(args.daemon and not args.foreground)
        log_path = log_file(log_dir, service_name)
        api_key = config.get("VLLM_API_KEY")
        proc = start_process(
            cmd,
            cwd=PROJECT_ROOT,
            log_path=log_path,
            daemon=daemon,
            env=serving_child_env(),
            secret=api_key,
        )
        write_pid(pid_file(pid_dir, service_name), proc.pid)
        LOGGER.info("Started %s pid=%s log=%s", service_name, proc.pid, log_path)
        timeout = float(config.get("VLLM_HEALTH_TIMEOUT") or default_native_health_timeout(None))
        result = wait_for_or_exit(
            proc,
            lambda: _health_ok(host, port, api_key),
            timeout=timeout,
            description=f"{service_name} /v1/models",
            log_path=log_path,
            secret=api_key,
        )
        if result != "ok":
            tail = log_tail(log_path, secret=api_key)
            hint = f"SGLang did not become healthy ({result}). Inspect {log_path}."
            if tail:
                hint += "\n--- log tail ---\n" + tail
            raise ConfigError(hint)
        if not daemon:
            return proc.wait()
        return 0
    except (ConfigError, VLLMLaunchError) as exc:
        LOGGER.error("%s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
