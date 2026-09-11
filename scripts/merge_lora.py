#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from common.bootstrap import PROJECT_ROOT, ensure_sys_path

ensure_sys_path()

from common.env import ConfigError, load_app_config, upsert_env_key
from common.logging_utils import setup_logging
from common.validate_model import looks_like_lora, validate_local_model

LOGGER = setup_logging("llm_tools.merge_lora")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge a LoRA adapter into the base model and export a full checkpoint.")
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--update-env", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config = load_app_config()
        base_model = Path(args.base_model) if args.base_model else config.require_path("BASE_MODEL")
        adapter = Path(args.adapter_path) if args.adapter_path else config.require_path("ADAPTER_PATH")
        output_dir = Path(args.output_dir) if args.output_dir else config.get_path("MERGED_MODEL_DIR")
        if output_dir is None:
            output_dir = PROJECT_ROOT / "training" / "output" / "merged"
        dtype = args.dtype or config.get("DTYPE") or "auto"

        base_problems = validate_local_model(base_model)
        if base_problems:
            raise ConfigError(f"Invalid base model {base_model}: {'; '.join(base_problems)}")
        if not looks_like_lora(adapter):
            raise ConfigError(f"Adapter path does not look like a LoRA directory: {adapter}")

        try:
            import torch
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            LOGGER.error("Missing training extras. Run `uv sync --extra train`. Import error: %s", exc)
            return 1

        torch_dtype = None
        if dtype in {"bf16", "bfloat16"}:
            torch_dtype = torch.bfloat16
        elif dtype in {"fp16", "float16"}:
            torch_dtype = torch.float16
        elif dtype in {"fp32", "float32"}:
            torch_dtype = torch.float32

        LOGGER.info("Loading base model from %s", base_model)
        model = AutoModelForCausalLM.from_pretrained(
            str(base_model),
            torch_dtype=torch_dtype or "auto",
            device_map="cpu",
            trust_remote_code=True,
        )
        LOGGER.info("Loading adapter from %s", adapter)
        model = PeftModel.from_pretrained(model, str(adapter))
        LOGGER.info("Merging LoRA weights")
        merged = model.merge_and_unload()
        output_dir.mkdir(parents=True, exist_ok=True)
        LOGGER.info("Saving merged model to %s", output_dir)
        merged.save_pretrained(str(output_dir), safe_serialization=True)
        tokenizer = AutoTokenizer.from_pretrained(str(base_model), trust_remote_code=True)
        tokenizer.save_pretrained(str(output_dir))

        problems = validate_local_model(output_dir)
        if problems:
            raise ConfigError(f"Merged model failed validation: {'; '.join(problems)}")

        if args.update_env:
            rel = output_dir
            try:
                rel_s = "./" + str(output_dir.resolve().relative_to(PROJECT_ROOT)).replace("\\", "/")
            except ValueError:
                rel_s = str(output_dir)
            upsert_env_key(config.env_path, "CHECKPOINT_PATH", rel_s)
            upsert_env_key(config.env_path, "MERGED_MODEL_DIR", rel_s)
            upsert_env_key(config.env_path, "TRAINED_MODEL_MODE", "merged")
            LOGGER.info("Updated CHECKPOINT_PATH / MERGED_MODEL_DIR / TRAINED_MODEL_MODE in .env")

        print(output_dir)
        return 0
    except ConfigError as exc:
        LOGGER.error("%s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
