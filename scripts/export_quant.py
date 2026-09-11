#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from common.bootstrap import PROJECT_ROOT, ensure_sys_path

ensure_sys_path()

from common.env import ConfigError, load_app_config, upsert_env_key
from common.logging_utils import setup_logging
from common.preflight import run_preflight
from common.validate_model import validate_local_model

LOGGER = setup_logging("llm_tools.export_quant")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a local HF model to AWQ / GPTQ / GGUF when converters are installed.")
    parser.add_argument("--method", choices=["awq", "gptq", "gguf"], required=True)
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--update-env", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _recipe(method: str) -> str:
    if method == "gguf":
        return (
            "GGUF conversion needs llama.cpp convert_hf_to_gguf.py or a compatible converter. "
            "Place it on PATH or install llama-cpp-python extras, then rerun."
        )
    if method == "awq":
        return "AWQ export needs `autoawq`. Install in a CUDA Linux env, then rerun this command."
    return "GPTQ export needs `gptqmodel` or `auto-gptq`. Install in a CUDA Linux env, then rerun this command."


def export_gguf(model_dir: Path, output_dir: Path, dry_run: bool) -> Path:
    converter = shutil.which("convert_hf_to_gguf.py") or shutil.which("convert-hf-to-gguf")
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / f"{model_dir.name}.gguf"
    if dry_run:
        LOGGER.info("Dry-run GGUF: converter=%s target=%s", converter, target)
        return target
    if not converter:
        raise ConfigError(_recipe("gguf"))
    import subprocess

    completed = subprocess.run([converter, str(model_dir), "--outfile", str(target)], check=False)
    if completed.returncode != 0 or not target.exists():
        raise ConfigError(f"GGUF converter failed with code {completed.returncode}. {_recipe('gguf')}")
    return target


def export_awq(model_dir: Path, output_dir: Path, dry_run: bool) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    if dry_run:
        return output_dir
    try:
        from awq import AutoAWQForCausalLM
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ConfigError(_recipe("awq")) from exc
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    model = AutoAWQForCausalLM.from_pretrained(str(model_dir), trust_remote_code=True)
    model.quantize(tokenizer, quant_config={"zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM"})
    model.save_quantized(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    return output_dir


def export_gptq(model_dir: Path, output_dir: Path, dry_run: bool) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    if dry_run:
        return output_dir
    try:
        from transformers import AutoTokenizer, GPTQConfig, AutoModelForCausalLM
    except Exception as exc:
        raise ConfigError(_recipe("gptq")) from exc
    try:
        quant = GPTQConfig(bits=4, dataset="c4", tokenizer=None)
    except Exception as exc:
        raise ConfigError(_recipe("gptq")) from exc
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    quant.tokenizer = tokenizer
    model = AutoModelForCausalLM.from_pretrained(str(model_dir), quantization_config=quant, trust_remote_code=True, device_map="auto")
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    return output_dir


def main() -> int:
    args = parse_args()
    try:
        config = load_app_config()
        model_dir = Path(args.model_dir) if args.model_dir else config.require_path("MODEL_DIR")
        problems = validate_local_model(model_dir)
        if problems and not args.dry_run:
            raise ConfigError("Refusing to export an incomplete model: " + "; ".join(problems))
        if problems:
            LOGGER.warning("Model looks incomplete (%s). Continuing because --dry-run was set.", "; ".join(problems))
        else:
            report = run_preflight(model_dir)
            for warning in report.warnings:
                LOGGER.warning("%s", warning)
        output_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "models" / f"{model_dir.name}-{args.method}"
        if not output_dir.is_absolute():
            output_dir = PROJECT_ROOT / output_dir
        LOGGER.info("Export method=%s model=%s output=%s", args.method, model_dir, output_dir)
        if args.method == "gguf":
            result = export_gguf(model_dir, output_dir, args.dry_run)
            quant_name = "gguf"
        elif args.method == "awq":
            result = export_awq(model_dir, output_dir, args.dry_run)
            quant_name = "awq"
        else:
            result = export_gptq(model_dir, output_dir, args.dry_run)
            quant_name = "gptq"
        manifest = {
            "method": args.method,
            "source": str(model_dir),
            "output": str(result),
            "dry_run": args.dry_run,
            "recipe": _recipe(args.method),
        }
        manifest_path = output_dir / "export_manifest.json"
        if not args.dry_run:
            output_dir.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if args.update_env and not args.dry_run:
            rel = result
            try:
                rel_path = Path(result).resolve().relative_to(PROJECT_ROOT)
                value = "./" + str(rel_path).replace("\\", "/")
            except ValueError:
                value = str(result)
            if result.is_dir():
                upsert_env_key(config.env_path, "MODEL_DIR", value)
            upsert_env_key(config.env_path, "QUANTIZATION", quant_name if args.method != "gguf" else "")
            LOGGER.info("Updated .env MODEL_DIR/QUANTIZATION from export.")
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return 0
    except (ConfigError, Exception) as exc:
        LOGGER.error("%s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
