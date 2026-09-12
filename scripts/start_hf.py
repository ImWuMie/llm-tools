#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from common.bootstrap import PROJECT_ROOT, ensure_sys_path

ensure_sys_path()

from common.env import ConfigError, apply_runtime_env, load_app_config
from common.hf_server import configure_runtime, load_causal_lm, serve_forever
from common.logging_utils import setup_logging
from common.preflight import run_preflight
from common.process import current_python, is_pid_running, log_file, pid_file, port_in_use, read_pid, start_process, wait_for_or_exit, write_pid
from common.secrets import redact_command
from common.health import check_openai_models
from common.validate_model import looks_like_lora

LOGGER = setup_logging("llm_tools.start_hf")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start an OpenAI-compatible transformers fallback server.")
    parser.add_argument("--foreground", action="store_true")
    parser.add_argument("--daemon", action="store_true")
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", default=None)
    parser.add_argument("--served-name", default=None)
    parser.add_argument("--service-name", default="hf")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def _health_ok(host: str, port: int, api_key: str | None) -> bool:
    try:
        check_openai_models(host, port, api_key=api_key, timeout=3)
        return True
    except Exception:
        return False


def main() -> int:
    args = parse_args()
    try:
        config = load_app_config()
        apply_runtime_env(config)
        model_dir = Path(args.model_dir) if args.model_dir else config.require_path("MODEL_DIR")
        adapter = Path(args.adapter_path) if args.adapter_path else None
        if adapter and not looks_like_lora(adapter):
            LOGGER.warning("Adapter path %s does not look like a LoRA directory.", adapter)
        host = args.host or config.require("VLLM_HOST")
        port = int(args.port or config.require("VLLM_PORT"))
        served = args.served_name or config.get("VLLM_SERVED_MODEL_NAME") or config.get("TRAINED_LORA_NAME") or model_dir.name
        api_key = config.get("VLLM_API_KEY")

        if args.worker:
            tokenizer, model = load_causal_lm(model_dir, adapter)
            configure_runtime(tokenizer=tokenizer, model=model, served_name=served, api_key=api_key)
            serve_forever(host, port)
            return 0

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
        if port_in_use(host, port):
            raise ConfigError(f"Port {port} is already in use on {host}.")

        pid_dir = config.require_path("PID_DIR")
        log_dir = config.require_path("LOG_DIR")
        service_name = args.service_name or "hf"
        existing = read_pid(pid_file(pid_dir, service_name))
        if existing and is_pid_running(existing):
            raise ConfigError(f"{service_name} already running as PID {existing}")

        cmd = [
            current_python(),
            str(PROJECT_ROOT / "scripts" / "start_hf.py"),
            "--worker",
            "--model-dir",
            str(model_dir),
            "--host",
            host,
            "--port",
            str(port),
            "--served-name",
            served,
        ]
        if adapter:
            cmd += ["--adapter-path", str(adapter)]
        LOGGER.info("HF command: %s", " ".join(redact_command(cmd)))
        daemon = bool(args.daemon and not args.foreground)
        log_path = log_file(log_dir, service_name)
        log_start = log_path.stat().st_size if log_path.is_file() else 0
        proc = start_process(cmd, cwd=PROJECT_ROOT, log_path=log_path, daemon=True)
        write_pid(pid_file(pid_dir, service_name), proc.pid)
        timeout = float(config.get("VLLM_HEALTH_TIMEOUT") or 180)
        result = wait_for_or_exit(
            proc,
            lambda: _health_ok(host, port, api_key),
            timeout=timeout,
            description="hf /v1/models",
            log_path=log_path,
            secret=api_key,
            log_start=log_start,
        )
        if result != "ok":
            raise ConfigError(f"transformers server started but never became healthy. Inspect {log_path}.")
        LOGGER.info("Started %s pid=%s log=%s", service_name, proc.pid, log_path)
        if not daemon:
            return proc.wait()
        return 0
    except (ConfigError, Exception) as exc:
        LOGGER.error("%s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
