#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
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
from common.windows_vllm_runtime import PYTORCH_CU130_INDEX

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
    parser.add_argument(
        "--install",
        action="store_true",
        help="Install a community Windows wheel from --wheel-url into this environment.",
    )
    parser.add_argument(
        "--wheel-url",
        default=None,
        help="Direct .whl URL from a vllm-windows GitHub release.",
    )
    parser.add_argument("--yes", action="store_true", help="Required to actually pip-install --wheel-url.")
    parser.add_argument(
        "--extra-index-url",
        default=PYTORCH_CU130_INDEX,
        help="PyTorch extra index so the wheel can resolve CUDA torch (default: cu130).",
    )
    return parser.parse_args()


COMMUNITY_RELEASES = (
    "https://github.com/SystemPanic/vllm-windows/releases",
    "https://github.com/devnen/vllm-windows/releases",
    "https://github.com/aivrar/vllm-windows-build/releases",
)


def _install_wheel(url: str, extra_index_url: str | None) -> int:
    uv = shutil.which("uv")
    extra: list[str] = []
    if extra_index_url:
        extra = ["--extra-index-url", extra_index_url]
        if uv:
            extra += ["--index-strategy", "unsafe-best-match"]
    if uv:
        cmd = [uv, "pip", "install", *extra, url]
    else:
        cmd = [sys.executable, "-m", "pip", "install", *extra, url]
    LOGGER.info("Installing community wheel: %s", url)
    completed = subprocess.run(cmd, check=False)
    return completed.returncode


def _write_native_env(config) -> None:
    upsert_env_key(config.env_path, "VLLM_WINDOWS_BACKEND", "native")
    if not (config.get("VLLM_USE_FLASHINFER_SAMPLER") or "").strip():
        upsert_env_key(config.env_path, "VLLM_USE_FLASHINFER_SAMPLER", "0")
    timeout = (config.get("VLLM_HEALTH_TIMEOUT") or "").strip()
    if timeout in {"", "180"}:
        upsert_env_key(config.env_path, "VLLM_HEALTH_TIMEOUT", "600")
    LOGGER.info(
        "Updated %s: VLLM_WINDOWS_BACKEND=native "
        "(FlashInfer sampler defaults off; health timeout bumped if it was 180s)",
        config.env_path,
    )


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
    print()
    print("Community release pages:")
    for url in COMMUNITY_RELEASES:
        print(f"  {url}")

    if not is_windows():
        LOGGER.info("This helper is for native Windows. Current platform=%s.", platform_name())
        if args.install:
            LOGGER.error("--install is only supported on native Windows.")
            return 1
        return 0

    if args.install:
        if not args.wheel_url:
            LOGGER.error(
                "--install requires --wheel-url pointing at a matching community .whl. "
                "Pick one from the release pages above. This repo will not auto-select a CUDA/Python wheel."
            )
            return 1
        if not args.yes:
            LOGGER.error("Refusing to install without --yes. Re-run with --install --wheel-url URL --yes.")
            return 1
        code = _install_wheel(args.wheel_url, args.extra_index_url)
        if code != 0:
            return code
        ok, detail = probe_vllm_import()
        LOGGER.info("After install: importable=%s %s", ok, detail)
        if not ok:
            return 1
        if args.write_env:
            config = load_app_config()
            _write_native_env(config)
        return 0

    if args.write_env:
        if not info["vllm_importable"]:
            LOGGER.error("Refusing to write VLLM_WINDOWS_BACKEND=native because `import vllm` failed.")
            return 1
        config = load_app_config()
        _write_native_env(config)

    if args.check and not info["vllm_importable"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
