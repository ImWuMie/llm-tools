#!/usr/bin/env python3
"""Launch vLLM's OpenAI server after applying toolchain runtime patches.

vLLM 0.29 + recent transformers bind ``validate_rope`` onto PreTrainedConfig
through HuggingFace Hub's ``@strict`` dataclass. Spark-X2.5 stores nested
per-attention ``rope_parameters``; transformers then injects a float
``rope_theta`` sibling and the bound validator crashes with
``'float' object has no attribute 'get'``.

Patching the module attribute is not enough. This wrapper patches the bound
class validators in the same process that then starts vLLM.
"""
from __future__ import annotations

import os
import runpy
import sys

from common.bootstrap import ensure_sys_path

ensure_sys_path()

from common.rope_compat import apply_all_patches  # noqa: E402


def main() -> None:
    apply_all_patches(log=True)
    if os.name == "nt":
        from common.windows_vllm_runtime import prepare_native_windows_interpreter

        prepare_native_windows_interpreter()
    sys.argv[0] = "vllm.entrypoints.openai.api_server"
    runpy.run_module("vllm.entrypoints.openai.api_server", run_name="__main__")


if __name__ == "__main__":
    main()
