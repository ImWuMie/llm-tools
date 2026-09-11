from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from .bootstrap import PROJECT_ROOT
from .env import AppConfig, ConfigError
from .health import check_openai_models
from .logging_utils import setup_logging
from .paths import to_wsl_path
from .platform_utils import has_docker, has_wsl, is_windows, vllm_native_supported, windows_vllm_hint
from .process import (
    current_python,
    is_pid_running,
    log_file,
    pid_file,
    port_in_use,
    read_pid,
    start_process,
    wait_for,
    write_pid,
)
from .secrets import redact_command
from .validate_model import looks_like_lora, looks_like_merged_model, validate_local_model

LOGGER = setup_logging("llm_tools.vllm")


class VLLMLaunchError(RuntimeError):
    pass


def _optional_flag(flag: str, value: str | None) -> list[str]:
    if value is None or str(value).strip() == "":
        return []
    return [flag, str(value)]


def build_vllm_command(
    python_executable: str,
    model_path: Path,
    config: AppConfig,
    extra_args: list[str] | None = None,
    lora_modules: dict[str, Path] | None = None,
) -> list[str]:
    host = config.require("VLLM_HOST")
    port = config.require("VLLM_PORT")
    cmd = [
        python_executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        str(model_path),
        "--host",
        host,
        "--port",
        str(port),
        "--tensor-parallel-size",
        str(config.get("TENSOR_PARALLEL_SIZE", "1")),
        "--gpu-memory-utilization",
        str(config.get("GPU_MEMORY_UTILIZATION", "0.9")),
        "--max-model-len",
        str(config.get("MAX_MODEL_LEN", "4096")),
        "--dtype",
        str(config.get("DTYPE", "auto")),
        "--max-num-seqs",
        str(config.get("VLLM_MAX_NUM_SEQS", "16")),
    ]
    api_key = config.get("VLLM_API_KEY")
    if api_key:
        cmd += ["--api-key", api_key]
    served = config.get("VLLM_SERVED_MODEL_NAME")
    if served:
        cmd += ["--served-model-name", served]
    cmd += _optional_flag("--quantization", config.get("QUANTIZATION"))
    if config.get_bool("VLLM_TRUST_REMOTE_CODE", True):
        cmd.append("--trust-remote-code")
    if lora_modules:
        cmd += ["--enable-lora", "--max-loras", str(max(1, len(lora_modules)))]
        for name, path in lora_modules.items():
            cmd += ["--lora-modules", f"{name}={path}"]
    if extra_args:
        cmd += extra_args
    return cmd


def detect_trained_mode(config: AppConfig, requested: str | None = None) -> tuple[str, Path, Path | None]:
    mode = (requested or config.get("TRAINED_MODEL_MODE") or "auto").strip().lower()
    checkpoint = config.get_path("CHECKPOINT_PATH")
    adapter = config.get_path("ADAPTER_PATH")
    base_model = config.get_path("BASE_MODEL") or config.get_path("MODEL_DIR")
    merged_dir = config.get_path("MERGED_MODEL_DIR")

    def pick_adapter() -> Path:
        for candidate in (adapter, checkpoint):
            if candidate and looks_like_lora(candidate):
                return candidate
        raise VLLMLaunchError(
            "TRAINED_MODEL_MODE=lora but no adapter_config.json was found. "
            "Set ADAPTER_PATH / CHECKPOINT_PATH to a LoRA directory, or merge first."
        )

    def pick_merged() -> Path:
        for candidate in (merged_dir, checkpoint, adapter):
            if candidate and looks_like_merged_model(candidate):
                return candidate
        raise VLLMLaunchError(
            "TRAINED_MODEL_MODE=merged but no full model directory was found. "
            "Run scripts/merge_lora.py or point CHECKPOINT_PATH at a merged export."
        )

    if mode == "lora":
        if base_model is None:
            raise VLLMLaunchError("BASE_MODEL is required when serving a LoRA adapter.")
        return "lora", base_model, pick_adapter()
    if mode == "merged":
        return "merged", pick_merged(), None
    if mode != "auto":
        raise VLLMLaunchError(f"Unknown TRAINED_MODEL_MODE={mode}. Use auto / lora / merged.")

    if adapter and looks_like_lora(adapter):
        if base_model is None:
            raise VLLMLaunchError("LoRA adapter detected but BASE_MODEL is missing.")
        return "lora", base_model, adapter
    if checkpoint and looks_like_lora(checkpoint):
        if base_model is None:
            raise VLLMLaunchError("LoRA checkpoint detected but BASE_MODEL is missing.")
        return "lora", base_model, checkpoint
    if merged_dir and looks_like_merged_model(merged_dir):
        return "merged", merged_dir, None
    if checkpoint and looks_like_merged_model(checkpoint):
        return "merged", checkpoint, None
    raise VLLMLaunchError(
        "Could not auto-detect trained model type. "
        "Set TRAINED_MODEL_MODE=lora|merged and verify CHECKPOINT_PATH / ADAPTER_PATH / MERGED_MODEL_DIR."
    )


def ensure_vllm_importable() -> None:
    try:
        import vllm  # noqa: F401
    except Exception as exc:  # pragma: no cover - depends on extra
        raise VLLMLaunchError(
            "vLLM is not installed in this environment. "
            "On Linux run `uv sync --extra infer`. On Windows use WSL2 or Docker.\n"
            f"Import error: {exc}"
        ) from exc


def vllm_supports_lora(python_executable: str) -> bool:
    probe = (
        "import inspect, sys\n"
        "try:\n"
        "    from vllm.engine.arg_utils import AsyncEngineArgs\n"
        "    names = set(inspect.signature(AsyncEngineArgs).parameters)\n"
        "    sys.exit(0 if 'enable_lora' in names else 1)\n"
        "except Exception:\n"
        "    sys.exit(1)\n"
    )
    completed = subprocess.run(
        [python_executable, "-c", probe],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    return completed.returncode == 0


def launch_via_wsl(args: list[str]) -> int:
    if not has_wsl():
        raise VLLMLaunchError("WSL was not found. Install WSL2 or use Docker.\n" + windows_vllm_hint())
    wsl = shutil.which("wsl")
    project = to_wsl_path(PROJECT_ROOT)
    inner = " ".join(["uv", "run", "python", *[f"'{item}'" if " " in item else item for item in args]])
    command = f"cd '{project}' && {inner}"
    LOGGER.info("Delegating to WSL2: wsl -e bash -lc <redacted project command>")
    completed = subprocess.run([wsl, "-e", "bash", "-lc", command], check=False)
    return completed.returncode


def launch_via_docker() -> None:
    if not has_docker():
        raise VLLMLaunchError("Docker was not found. Install Docker Desktop / Engine, or use WSL2.\n" + windows_vllm_hint())
    LOGGER.info("Starting docker compose service `vllm`.")
    compose = shutil.which("docker")
    completed = subprocess.run([compose, "compose", "up", "-d", "vllm"], cwd=str(PROJECT_ROOT), check=False)
    if completed.returncode != 0:
        raise VLLMLaunchError("docker compose up vllm failed. Check Docker GPU runtime and .env.")


def start_vllm_server(
    config: AppConfig,
    model_path: Path,
    service_name: str,
    daemon: bool,
    extra_args: list[str] | None = None,
    lora_modules: dict[str, Path] | None = None,
    windows_backend: str | None = None,
    force_native: bool = False,
) -> int:
    problems = validate_local_model(model_path)
    if problems and not lora_modules:
        pretty = "; ".join(problems)
        raise VLLMLaunchError(
            f"Model directory {model_path} failed validation: {pretty}. "
            "Download the model first with scripts/download_model.py."
        )
    if lora_modules:
        base_problems = validate_local_model(model_path)
        if base_problems:
            raise VLLMLaunchError(
                f"BASE_MODEL {model_path} is invalid: {'; '.join(base_problems)}"
            )

    host = config.require("VLLM_HOST")
    port = int(config.require("VLLM_PORT"))
    if port_in_use(host, port):
        raise VLLMLaunchError(
            f"Port {port} is already in use on {host}. "
            "Stop the existing server with `uv run python scripts/stop_vllm.py` "
            "or change VLLM_PORT in .env."
        )

    pid_dir = config.require_path("PID_DIR")
    log_dir = config.require_path("LOG_DIR")
    existing = read_pid(pid_file(pid_dir, service_name))
    if existing and is_pid_running(existing):
        raise VLLMLaunchError(f"{service_name} already running as PID {existing}. Use stop_vllm.py first.")

    backend = (windows_backend or config.get("VLLM_WINDOWS_BACKEND") or "wsl").strip().lower()
    if is_windows() and not vllm_native_supported() and not force_native:
        LOGGER.warning(windows_vllm_hint())
        if backend == "docker":
            launch_via_docker()
            return 0
        if backend == "wsl":
            script = "scripts/start_vllm.py" if service_name == "vllm" else "scripts/start_vllm_trained.py"
            forwarded = [script, "--foreground" if not daemon else "--daemon", "--force-native"]
            return launch_via_wsl(forwarded)
        raise VLLMLaunchError(windows_vllm_hint())

    ensure_vllm_importable()
    python_executable = current_python()
    if lora_modules and not vllm_supports_lora(python_executable):
        raise VLLMLaunchError(
            "This vLLM build does not expose --enable-lora. "
            "Merge the adapter first: uv run python scripts/merge_lora.py"
        )

    cmd = build_vllm_command(
        python_executable=python_executable,
        model_path=model_path,
        config=config,
        extra_args=extra_args,
        lora_modules=lora_modules,
    )
    LOGGER.info("vLLM command: %s", " ".join(redact_command(cmd)))
    log_path = log_file(log_dir, service_name)
    proc = start_process(cmd, cwd=PROJECT_ROOT, log_path=log_path, daemon=daemon)
    write_pid(pid_file(pid_dir, service_name), proc.pid)
    LOGGER.info("Started %s pid=%s log=%s", service_name, proc.pid, log_path)

    timeout = float(config.get("VLLM_HEALTH_TIMEOUT") or 180)
    api_key = config.get("VLLM_API_KEY")
    healthy = wait_for(
        lambda: _health_ok(host, port, api_key),
        timeout=timeout,
        description=f"{service_name} /v1/models",
    )
    if not healthy:
        hint = (
            "The process started but never became healthy. Typical causes: GPU OOM, "
            "invalid MODEL_DIR, missing CUDA, or quantization mismatch. "
            f"Inspect {log_path}."
        )
        if not daemon:
            proc.terminate()
        raise VLLMLaunchError(hint)
    if not daemon:
        return proc.wait()
    return 0


def _health_ok(host: str, port: int, api_key: str | None) -> bool:
    try:
        check_openai_models(host, port, api_key=api_key, timeout=3)
        return True
    except Exception:
        return False
