from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


def is_windows() -> bool:
    return os.name == "nt" or sys.platform.startswith("win")


def is_linux() -> bool:
    return sys.platform.startswith("linux")


def is_wsl() -> bool:
    if not is_linux():
        return False
    try:
        text = Path("/proc/version").read_text(encoding="utf-8", errors="ignore").lower()
    except OSError:
        return False
    return "microsoft" in text


def has_wsl() -> bool:
    return is_windows() and shutil.which("wsl") is not None


def has_docker() -> bool:
    return shutil.which("docker") is not None


def cuda_visible() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def platform_name() -> str:
    if is_wsl():
        return "wsl"
    if is_windows():
        return "windows"
    if is_linux():
        return "linux"
    return sys.platform


def vllm_native_supported() -> bool:
    return is_linux()


def windows_vllm_hint() -> str:
    lines = [
        "vLLM does not officially support native Windows.",
        "Use one of these options:",
        "  1) WSL2:  wsl -e bash -lc 'cd /mnt/<drive>/path/to/llm-tools && uv run python scripts/start_vllm.py --daemon'",
        "  2) Docker: docker compose up vllm",
        "  3) Re-run with --wsl if WSL2 is installed, or set VLLM_WINDOWS_BACKEND=wsl|docker",
    ]
    if has_wsl():
        lines.append("WSL executable was detected on this machine.")
    if has_docker():
        lines.append("Docker executable was detected on this machine.")
    return "\n".join(lines)
