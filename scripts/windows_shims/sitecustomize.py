"""Injected via PYTHONPATH so native Windows vLLM children pick up DLL/PATH fixes."""

from __future__ import annotations


def _run() -> None:
    try:
        from common.windows_vllm_runtime import prepare_native_windows_interpreter

        prepare_native_windows_interpreter()
        return
    except Exception:
        pass
    import os
    import sys
    from pathlib import Path

    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    scripts = Path(sys.executable).resolve().parent
    os.environ["PATH"] = str(scripts) + os.pathsep + os.environ.get("PATH", "")
    adder = getattr(os, "add_dll_directory", None)
    if adder is not None:
        try:
            adder(str(scripts))
        except OSError:
            pass


_run()
