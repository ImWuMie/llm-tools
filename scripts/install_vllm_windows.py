#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys

from common.bootstrap import ensure_sys_path

ensure_sys_path()

from common.env import ConfigError, load_app_config, upsert_env_key
from common.logging_utils import setup_logging
from common.platform_utils import (
    cuda_visible,
    has_docker,
    has_wsl,
    is_windows,
    native_windows_vllm_guide,
    platform_name,
    probe_vllm_import,
    resolve_windows_vllm_backend,
    windows_vllm_hint,
)

LOGGER = setup_logging("llm_tools.install_vllm_windows")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check or enable the optional unofficial native Windows vLLM path. "
            "This does not download community wheels automatically."
        )
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Probe this Python env and print the resolved Windows backend.",
    )
    parser.add_argument(
        "--write-env",
        action="store_true",
        help="If native import works, set VLLM_WINDOWS_BACKEND=native in .env.",
    )
    return parser.parse_args()


def _python_info() -> dict[str, object]:
    torch_detail = "not installed"
    try:
        import torch

        cuda = bool(getattr(torch, "cuda", None) and torch.cuda.is_available())
        torch_detail = {
            "version": getattr(torch, "__version__", "unknown"),
            "cuda_available": cuda,
            "cuda_version": getattr(getattr(torch, "version", None), "cuda", None),
        }
    except Exception as exc:
        torch_detail = f"{type(exc).__name__}: {exc}"
    importable, detail = probe_vllm_import()
    return {
        "platform": platform_name(),
        "windows": is_windows(),
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "wsl": has_wsl(),
        "docker": has_docker(),
        "torch": torch_detail,
        "cuda_visible": cuda_visible(),
        "vllm_importable": importable,
        "vllm_detail": detail,
    }


def main() -> int:
    args = parse_args()
    info = _python_info()
    requested = "auto"
    env_path = None
    try:
        config = load_app_config()
        requested = config.get("VLLM_WINDOWS_BACKEND") or "wsl"
        env_path = str(config.env_path)
    except ConfigError as exc:
        LOGGER.warning("%s", exc)

    resolved = resolve_windows_vllm_backend(
        requested,
        vllm_importable=bool(info["vllm_importable"]),
        wsl_available=bool(info["wsl"]),
        docker_available=bool(info["docker"]),
    )
    payload = {
        **info,
        "env_path": env_path,
        "requested_backend": requested,
        "resolved_backend": resolved,
        "official_native_windows": False,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print()
    print(windows_vllm_hint())
    print()
    print(native_windows_vllm_guide())

    if not is_windows():
        LOGGER.info("This helper is for native Windows. Current platform=%s.", platform_name())
        return 0

    if args.write_env:
        if not info["vllm_importable"]:
            LOGGER.error("Refusing to write VLLM_WINDOWS_BACKEND=native because `import vllm` failed.")
            return 1
        config = load_app_config()
        upsert_env_key(config.env_path, "VLLM_WINDOWS_BACKEND", "native")
        LOGGER.info("Updated %s: VLLM_WINDOWS_BACKEND=native", config.env_path)

    if args.check and not info["vllm_importable"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
