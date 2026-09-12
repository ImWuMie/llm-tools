"""Loaded via PYTHONPATH so vLLM/HF children pick up runtime patches."""

from __future__ import annotations

import os


def _run() -> None:
    try:
        from common.rope_compat import patch_transformers_rope

        patch_transformers_rope()
    except Exception:
        pass
    if os.name == "nt":
        try:
            from common.windows_vllm_runtime import prepare_native_windows_interpreter

            prepare_native_windows_interpreter()
            return
        except Exception:
            pass


_run()
