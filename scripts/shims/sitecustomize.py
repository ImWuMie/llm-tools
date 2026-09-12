"""Loaded via PYTHONPATH so vLLM/HF children pick up runtime patches."""

from __future__ import annotations

import os
import sys


def _run() -> None:
    try:
        from common.rope_compat import apply_all_patches

        apply_all_patches(log=True)
    except Exception as exc:
        sys.stderr.write(f"llm-tools: sitecustomize RoPE patch failed: {type(exc).__name__}: {exc}\n")
    if os.name == "nt":
        try:
            from common.windows_vllm_runtime import prepare_native_windows_interpreter

            prepare_native_windows_interpreter()
        except Exception as exc:
            sys.stderr.write(
                f"llm-tools: sitecustomize Windows runtime failed: {type(exc).__name__}: {exc}\n"
            )


_run()
