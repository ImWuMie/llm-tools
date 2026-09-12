from __future__ import annotations

import os
import sys
import types
from collections.abc import Mapping
from pathlib import Path

from .bootstrap import PROJECT_ROOT, SCRIPTS_DIR

# Keys that belong to this toolchain, not to upstream vLLM. Forwarding them
# makes vLLM 0.26+ log "Unknown vLLM environment variable".
TOOLCHAIN_ONLY_ENV = {
    "VLLM_WINDOWS_BACKEND",
    "VLLM_HEALTH_TIMEOUT",
    "VLLM_TRUST_REMOTE_CODE",
    "VLLM_MAX_NUM_SEQS",
    "VLLM_SERVED_MODEL_NAME",
    "VLLM_HOST",
    "VLLM_PORT",
    "VLLM_SPARK_PLUGIN",
    "VLLM_TOOL_CALL_PARSER",
    "VLLM_ENABLE_AUTO_TOOL_CHOICE",
}

PYTORCH_CU130_INDEX = "https://download.pytorch.org/whl/cu130"
WINDOWS_SHIMS_DIR = SCRIPTS_DIR / "windows_shims"
TVM_FFI_LIB_ENV = "LLM_TOOLS_TVM_FFI_LIB"


def prepend_path(env: dict[str, str], directory: Path) -> None:
    if not directory.exists():
        return
    prefix = str(directory)
    current = env.get("PATH", "")
    parts = current.split(os.pathsep) if current else []
    if prefix not in parts:
        env["PATH"] = prefix + (os.pathsep + current if current else "")


def tvm_ffi_lib_dir() -> Path | None:
    explicit = os.environ.get(TVM_FFI_LIB_ENV)
    if explicit:
        path = Path(explicit)
        if path.is_dir():
            return path
    try:
        import tvm_ffi
    except Exception:
        return None
    path = Path(tvm_ffi.__file__).resolve().parent / "lib"
    return path if path.is_dir() else None


def add_dll_directory(path: Path) -> None:
    if not path.is_dir():
        return
    adder = getattr(os, "add_dll_directory", None)
    if adder is not None:
        try:
            adder(str(path))
        except OSError:
            pass
    prepend_path(os.environ, path)


def install_xgrammar_stub(reason: str) -> None:
    existing = sys.modules.get("xgrammar")
    if existing is not None and not getattr(existing, "_llm_tools_stub", False):
        return
    stub = types.ModuleType("xgrammar")
    stub._llm_tools_stub = True  # type: ignore[attr-defined]
    stub._llm_tools_stub_reason = reason  # type: ignore[attr-defined]

    class _Missing:
        def __init__(self, *args: object, **kwargs: object) -> None:
            raise RuntimeError(
                "xgrammar native bindings failed to load on this Windows vLLM wheel. "
                f"Structured output is disabled ({reason}). Regular chat still works."
            )

        def __getattr__(self, name: str) -> object:
            raise RuntimeError(
                "xgrammar native bindings failed to load on this Windows vLLM wheel. "
                f"Structured output is disabled ({reason})."
            )

    class GrammarMatcher:
        """Placeholder so vLLM can import XgrammarGrammar annotations."""

    stub.GrammarMatcher = GrammarMatcher  # type: ignore[attr-defined]
    stub.TokenizerInfo = _Missing  # type: ignore[attr-defined]
    stub.GrammarCompiler = _Missing  # type: ignore[attr-defined]
    stub.CompiledGrammar = _Missing  # type: ignore[attr-defined]
    sys.modules["xgrammar"] = stub
    sys.stderr.write(
        f"llm-tools: xgrammar native bindings unavailable ({reason}); "
        "structured output disabled, chat serving continues.\n"
    )


def prepare_native_windows_interpreter() -> None:
    """Fix DLL search, ninja PATH, FlashInfer sampler, and xgrammar import.

    Safe to call from sitecustomize in the vLLM child process.
    """
    scripts_dir = Path(sys.executable).resolve().parent
    add_dll_directory(scripts_dir)
    lib = tvm_ffi_lib_dir()
    if lib is not None:
        add_dll_directory(lib)
        os.environ[TVM_FFI_LIB_ENV] = str(lib)
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    if "xgrammar" in sys.modules:
        return
    try:
        import xgrammar  # noqa: F401
    except Exception as exc:
        install_xgrammar_stub(f"{type(exc).__name__}: {exc}")


def native_windows_child_env(base: Mapping[str, str] | None = None) -> dict[str, str]:
    """Environment for `python -m vllm.entrypoints.openai.api_server` on Windows."""
    env = dict(base or os.environ)
    prepend_path(env, Path(sys.executable).resolve().parent)
    lib = tvm_ffi_lib_dir()
    if lib is not None:
        prepend_path(env, lib)
        env[TVM_FFI_LIB_ENV] = str(lib)
    env.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    pythonpath = [
        str(WINDOWS_SHIMS_DIR),
        str(SCRIPTS_DIR),
        str(PROJECT_ROOT),
    ]
    existing = env.get("PYTHONPATH", "")
    if existing:
        pythonpath.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)
    for key in TOOLCHAIN_ONLY_ENV:
        env.pop(key, None)
    return env


def serving_child_env(base: Mapping[str, str] | None = None) -> dict[str, str]:
    """Environment for vLLM/HF child processes on any platform."""
    if os.name == "nt":
        return native_windows_child_env(base)
    env = dict(base or os.environ)
    for key in TOOLCHAIN_ONLY_ENV:
        env.pop(key, None)
    return env


def default_native_health_timeout(configured: float | None) -> float:
    if configured is None:
        return 600.0
    return float(configured)


def log_tail(path: Path, max_chars: int = 4000, secret: str | None = None) -> str:
    if not path.is_file():
        return ""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    if secret:
        text = text.replace(secret, "***")
    return text[-max_chars:]
