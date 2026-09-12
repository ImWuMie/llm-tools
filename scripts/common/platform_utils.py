from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

WINDOWS_VLLM_BACKENDS = ("auto", "wsl", "docker", "native", "fail")


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


def probe_vllm_import() -> tuple[bool, str]:
    """Return whether `vllm` imports in this interpreter, plus a short detail string."""
    try:
        import vllm

        version = getattr(vllm, "__version__", "unknown")
        return True, f"vllm {version}"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def resolve_windows_vllm_backend(
    requested: str | None,
    *,
    force_native: bool = False,
    vllm_importable: bool | None = None,
    wsl_available: bool | None = None,
    docker_available: bool | None = None,
) -> str:
    """Resolve the Windows serving backend.

    `auto` prefers an already-installed community Windows wheel, then WSL2, then Docker.
    """
    if force_native:
        return "native"
    backend = (requested or "wsl").strip().lower() or "wsl"
    if backend not in WINDOWS_VLLM_BACKENDS:
        raise ValueError(
            f"Unknown VLLM_WINDOWS_BACKEND={backend}. "
            "Use auto / wsl / docker / native / fail."
        )
    if backend != "auto":
        return backend
    if vllm_importable is None:
        vllm_importable = probe_vllm_import()[0]
    if vllm_importable:
        return "native"
    if wsl_available is None:
        wsl_available = has_wsl()
    if wsl_available:
        return "wsl"
    if docker_available is None:
        docker_available = has_docker()
    if docker_available:
        return "docker"
    return "fail"


def native_windows_vllm_guide() -> str:
    return "\n".join(
        [
            "Official vLLM does not publish Windows wheels.",
            "Optional unofficial native path (community vllm-windows):",
            "  1) Use a separate Python 3.12 env that matches the wheel's CUDA (do not mix with Linux infer extra).",
            "  2) Install a matching wheel (Python 3.12 / CUDA 13 / Blackwell = cu132) with:",
            "       uv run python scripts/install_vllm_windows.py --install --wheel-url URL --yes --write-env",
            "     The helper adds PyTorch cu130 extra index automatically.",
            "  3) Confirm:  uv run python scripts/install_vllm_windows.py --check",
            "  4) Start:    uv run python scripts/start_vllm.py --daemon --native",
            "     or set    VLLM_WINDOWS_BACKEND=native   (or auto)",
            "Native start injects ninja PATH, disables FlashInfer sampler JIT by default,",
            "and stubs xgrammar if the community DLL fails to load. Custom architectures",
            "(Spark2_5ForCausalLM) still need --engine hf.",
            "This path is unsupported. WSL2 / Docker remain the recommended backends.",
        ]
    )


def windows_vllm_hint() -> str:
    lines = [
        "vLLM does not officially support native Windows.",
        "Use one of these options:",
        "  1) WSL2:  wsl -e bash -lc 'cd /mnt/<drive>/path/to/llm-tools && uv run python scripts/start_vllm.py --daemon'",
        "  2) Docker: docker compose up vllm",
        "  3) Optional native: install a community vllm-windows wheel, then --native or VLLM_WINDOWS_BACKEND=native|auto",
        "  4) Inspect the current env: uv run python scripts/install_vllm_windows.py --check",
    ]
    if has_wsl():
        lines.append("WSL executable was detected on this machine.")
    if has_docker():
        lines.append("Docker executable was detected on this machine.")
    return "\n".join(lines)
