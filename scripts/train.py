#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path
from typing import Any

from common.bootstrap import PROJECT_ROOT, ensure_sys_path

ensure_sys_path()

from common.data_format import DataFormatError, DataOptions
from common.env import ConfigError, load_app_config, upsert_env_key
from common.logging_utils import setup_logging
from common.paths import resolve_path
from common.platform_utils import is_windows
from common.validate_model import validate_local_model
from common.hub_data import materialize_training_data
from common.schema import raise_if_errors, validate_train_config

LOGGER = setup_logging("llm_tools.train")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a local model with LoRA / QLoRA.")
    parser.add_argument("--config", default=None, help="Path to training/config.json")
    parser.add_argument("--data", required=True, help="Training file path")
    parser.add_argument(
        "--data-format",
        default="default",
        choices=["default", "txt", "jsonl", "sharegpt", "alpaca", "custom"],
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--update-env", action="store_true")
    parser.add_argument("--data-source", choices=["auto", "local", "hf", "modelscope"], default="auto")
    parser.add_argument("--data-split", default="train")
    parser.add_argument("--eval-data", default=None, help="Optional held-out eval file (same --data-format).")
    parser.add_argument("--eval-split", type=float, default=None, help="Fraction of train data used for eval_loss. Default from config (0.1).")
    return parser.parse_args()


def load_train_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"Training config not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ConfigError("training config must be a JSON object")
    return payload


def data_options_from_config(cfg: dict[str, Any], system_prompt: str) -> DataOptions:
    data_cfg = cfg.get("data") or {}
    custom = cfg.get("custom_format") or {}
    return DataOptions(
        skip_empty_lines=bool(data_cfg.get("skip_empty_lines", True)),
        skip_comment_lines=bool(data_cfg.get("skip_comment_lines", True)),
        comment_prefix=str(data_cfg.get("comment_prefix", "#")),
        encoding=str(data_cfg.get("encoding", "utf-8")),
        system_prompt=system_prompt,
        custom_user_key=str(custom.get("user_key", "user")),
        custom_assistant_key=str(custom.get("assistant_key", "assistant")),
        custom_system_key=str(custom.get("system_key", "system")),
    )


def render_text(messages: list[dict[str, str]], tokenizer: Any) -> str:
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    parts: list[str] = []
    for item in messages:
        parts.append(f"<|im_start|>{item['role']}\n{item['content']}<|im_end|>")
    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)


def latest_checkpoint(output_dir: Path) -> Path | None:
    checkpoints = sorted(output_dir.glob("checkpoint-*"), key=lambda p: p.stat().st_mtime)
    return checkpoints[-1] if checkpoints else None


def split_records(records: list[dict[str, Any]], eval_split: float, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Hold out a shuffle split for HuggingFace-style eval_loss."""
    import random

    n = len(records)
    if eval_split <= 0 or n < 2:
        return records, []
    n_eval = max(1, int(round(n * float(eval_split))))
    n_eval = min(n_eval, n - 1)
    indices = list(range(n))
    random.Random(int(seed)).shuffle(indices)
    eval_ids = set(indices[:n_eval])
    train_records = [records[i] for i in range(n) if i not in eval_ids]
    eval_records = [records[i] for i in range(n) if i in eval_ids]
    return train_records, eval_records


def resolve_eval_strategy(training_cfg: dict[str, Any], *, has_eval: bool) -> tuple[str, int]:
    if not has_eval:
        return "no", 0
    strategy = str(
        training_cfg.get("eval_strategy")
        or training_cfg.get("evaluation_strategy")
        or "steps"
    ).strip().lower()
    if strategy in {"", "none", "no", "false"}:
        return "no", 0
    if strategy not in {"steps", "epoch"}:
        strategy = "steps"
    eval_steps = int(training_cfg.get("eval_steps") or training_cfg.get("logging_steps") or 10)
    return strategy, max(1, eval_steps)


def build_trainer(
    model,
    tokenizer,
    dataset,
    training_cfg: dict[str, Any],
    output_dir: Path,
    lora_cfg,
    max_seq_length: int,
    eval_dataset=None,
):
    from transformers import TrainerCallback, TrainingArguments
    from trl import SFTTrainer

    class GpuMemoryCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            try:
                import torch

                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated() / 1024**3
                    reserved = torch.cuda.memory_reserved() / 1024**3
                    LOGGER.info(
                        "step=%s loss=%s eval_loss=%s lr=%s gpu_alloc=%.2fGiB gpu_reserved=%.2fGiB",
                        state.global_step,
                        None if not logs else logs.get("loss"),
                        None if not logs else logs.get("eval_loss"),
                        None if not logs else logs.get("learning_rate"),
                        allocated,
                        reserved,
                    )
                elif logs and "eval_loss" in logs:
                    LOGGER.info(
                        "step=%s loss=%s eval_loss=%s lr=%s",
                        state.global_step,
                        logs.get("loss"),
                        logs.get("eval_loss"),
                        logs.get("learning_rate"),
                    )
            except Exception:
                return

    has_eval = eval_dataset is not None
    eval_strategy, eval_steps = resolve_eval_strategy(training_cfg, has_eval=has_eval)
    args_kwargs = dict(
        output_dir=str(output_dir),
        num_train_epochs=float(training_cfg.get("num_train_epochs", 3)),
        learning_rate=float(training_cfg.get("learning_rate", 2e-4)),
        per_device_train_batch_size=int(training_cfg.get("per_device_train_batch_size", 1)),
        gradient_accumulation_steps=int(training_cfg.get("gradient_accumulation_steps", 8)),
        logging_steps=int(training_cfg.get("logging_steps", 10)),
        save_steps=int(training_cfg.get("save_steps", 100)),
        save_total_limit=int(training_cfg.get("save_total_limit", 3)),
        seed=int(training_cfg.get("seed", 42)),
        fp16=bool(training_cfg.get("fp16", False)),
        bf16=bool(training_cfg.get("bf16", True)),
        gradient_checkpointing=bool(training_cfg.get("gradient_checkpointing", True)),
        report_to=training_cfg.get("report_to") or "none",
        remove_unused_columns=False,
        lr_scheduler_type=str(training_cfg.get("lr_scheduler_type", "cosine")),
        warmup_ratio=float(training_cfg.get("warmup_ratio", 0.03)),
        logging_first_step=True,
        eval_strategy=eval_strategy,
        evaluation_strategy=eval_strategy,
        eval_steps=eval_steps,
        per_device_eval_batch_size=int(
            training_cfg.get("per_device_eval_batch_size") or training_cfg.get("per_device_train_batch_size") or 1
        ),
        do_eval=has_eval and eval_strategy != "no",
        load_best_model_at_end=bool(training_cfg.get("load_best_model_at_end", False)) and has_eval and eval_strategy != "no",
        metric_for_best_model=str(training_cfg.get("metric_for_best_model") or "eval_loss"),
        greater_is_better=bool(training_cfg.get("greater_is_better", False)),
    )
    if eval_strategy != "steps":
        args_kwargs.pop("eval_steps", None)

    trainer_kwargs_list: list[dict[str, Any]] = []
    try:
        from trl import SFTConfig

        sft_kwargs = dict(args_kwargs)
        sft_kwargs["max_seq_length"] = max_seq_length
        sft_kwargs["dataset_text_field"] = "text"
        sft_kwargs["packing"] = False
        sft_args = SFTConfig(**_filter_kwargs(SFTConfig, sft_kwargs))
        trainer_kwargs_list.append(
            {
                "model": model,
                "args": sft_args,
                "train_dataset": dataset,
                "processing_class": tokenizer,
                "peft_config": lora_cfg,
            }
        )
        trainer_kwargs_list.append(
            {
                "model": model,
                "args": sft_args,
                "train_dataset": dataset,
                "tokenizer": tokenizer,
                "peft_config": lora_cfg,
            }
        )
    except Exception as exc:
        LOGGER.info("SFTConfig not available or rejected kwargs (%s); falling back to TrainingArguments.", exc)

    training_args = TrainingArguments(**_filter_kwargs(TrainingArguments, args_kwargs))
    trainer_kwargs_list.extend(
        [
            {
                "model": model,
                "args": training_args,
                "train_dataset": dataset,
                "processing_class": tokenizer,
                "peft_config": lora_cfg,
                "dataset_text_field": "text",
                "max_seq_length": max_seq_length,
            },
            {
                "model": model,
                "args": training_args,
                "train_dataset": dataset,
                "tokenizer": tokenizer,
                "peft_config": lora_cfg,
                "dataset_text_field": "text",
                "max_seq_length": max_seq_length,
            },
        ]
    )

    last_error: Exception | None = None
    for kwargs in trainer_kwargs_list:
        if eval_dataset is not None:
            kwargs = dict(kwargs)
            kwargs["eval_dataset"] = eval_dataset
        try:
            trainer = SFTTrainer(**_filter_kwargs(SFTTrainer, kwargs))
            trainer.add_callback(GpuMemoryCallback())
            return trainer
        except TypeError as exc:
            last_error = exc
            continue
    raise ConfigError(f"Unable to construct SFTTrainer with this trl version: {last_error}")


def _filter_kwargs(cls, kwargs: dict[str, Any]) -> dict[str, Any]:
    try:
        params = inspect.signature(cls).parameters
    except (TypeError, ValueError):
        return kwargs
    if any(item.kind == inspect.Parameter.VAR_KEYWORD for item in params.values()):
        return kwargs
    return {key: value for key, value in kwargs.items() if key in params}


def load_model_and_tokenizer(base_model: Path, cfg: dict[str, Any]):
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    quant = cfg.get("quantization") or {}
    load_in_4bit = bool(quant.get("load_in_4bit", False))
    lora = cfg.get("lora") or {}
    torch_dtype = torch.bfloat16 if cfg.get("bf16", True) else (torch.float16 if cfg.get("fp16") else torch.float32)

    tokenizer = AutoTokenizer.from_pretrained(str(base_model), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": torch_dtype,
    }
    bnb_config = None
    if load_in_4bit:
        if is_windows():
            LOGGER.warning("QLoRA/bitsandbytes is unreliable on native Windows. Falling back to LoRA without 4-bit.")
            load_in_4bit = False
        else:
            try:
                import bitsandbytes  # noqa: F401

                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type=str(quant.get("bnb_4bit_quant_type", "nf4")),
                    bnb_4bit_compute_dtype=torch_dtype,
                    bnb_4bit_use_double_quant=bool(quant.get("bnb_4bit_use_double_quant", True)),
                )
                model_kwargs["quantization_config"] = bnb_config
                model_kwargs["device_map"] = "auto"
            except Exception as exc:
                LOGGER.warning("bitsandbytes unavailable (%s). Falling back to LoRA without 4-bit.", exc)
                load_in_4bit = False

    if not load_in_4bit:
        if torch.cuda.is_available():
            model_kwargs["device_map"] = "auto"

    LOGGER.info("Loading base model from %s (4bit=%s)", base_model, load_in_4bit)
    try:
        model = AutoModelForCausalLM.from_pretrained(str(base_model), **model_kwargs)
    except torch.cuda.OutOfMemoryError as exc:
        raise ConfigError(
            "GPU out of memory while loading the model. Try enabling quantization.load_in_4bit, "
            "reducing max_seq_length, or using a smaller base model."
        ) from exc

    if bool(cfg.get("gradient_checkpointing", True)) and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    peft_config = LoraConfig(
        r=int(lora.get("r", 8)),
        lora_alpha=int(lora.get("alpha", 16)),
        lora_dropout=float(lora.get("dropout", 0.05)),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=lora.get("target_modules") or ["q_proj", "v_proj"],
    )
    if load_in_4bit:
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, peft_config)
        return model, tokenizer, None
    return model, tokenizer, peft_config


def main() -> int:
    args = parse_args()
    try:
        config = load_app_config()
        train_cfg_path = Path(args.config) if args.config else PROJECT_ROOT / "training" / "config.json"
        train_cfg = load_train_config(train_cfg_path)
        schema_problems = validate_train_config(train_cfg, strict=config.get_bool("CONFIG_STRICT", False))
        for item in schema_problems:
            if item.startswith("warning:"):
                LOGGER.warning("%s", item)
            else:
                LOGGER.error("%s", item)
        raise_if_errors(schema_problems, label=str(train_cfg_path))

        base_model = resolve_path(train_cfg.get("base_model"), PROJECT_ROOT) or config.get_path("BASE_MODEL")
        if base_model is None:
            raise ConfigError("base_model is missing in training/config.json and BASE_MODEL is empty.")
        problems = validate_local_model(base_model)
        if problems:
            LOGGER.warning(
                "Base model at %s looks incomplete (%s). Training may still work for tiny test models, "
                "but you should download a full snapshot for real runs.",
                base_model,
                "; ".join(problems),
            )

        output_dir = Path(args.output_dir) if args.output_dir else resolve_path(train_cfg.get("output_dir"), PROJECT_ROOT)
        if output_dir is None:
            output_dir = PROJECT_ROOT / "training" / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        system_prompt_path = PROJECT_ROOT / "training" / "system_prompt.txt"
        system_prompt = ""
        if system_prompt_path.exists():
            system_prompt = system_prompt_path.read_text(encoding="utf-8").strip()
        if system_prompt:
            LOGGER.info("Using system prompt from %s (%s chars)", system_prompt_path, len(system_prompt))
        else:
            LOGGER.info("system_prompt.txt is empty; no system message will be added.")

        options = data_options_from_config(train_cfg, system_prompt)
        try:
            samples, processed_path = materialize_training_data(
                args.data,
                data_format=args.data_format,
                options=options,
                project_root=PROJECT_ROOT,
                source=args.data_source,
                split=args.data_split,
                hf_token=config.get("HF_TOKEN"),
            )
        except DataFormatError as exc:
            raise ConfigError(str(exc)) from exc
        LOGGER.info("Using processed dataset %s (%s samples)", processed_path, len(samples))

        try:
            from datasets import Dataset
            from transformers import AutoTokenizer
        except ImportError as exc:
            LOGGER.error("Training extras are missing. Run `uv sync --extra train`. Import error: %s", exc)
            return 1

        tokenizer = AutoTokenizer.from_pretrained(str(base_model), trust_remote_code=True)
        records = [{"messages": messages, "text": render_text(messages, tokenizer)} for messages in samples]
        eval_records: list[dict[str, Any]] = []
        eval_data = args.eval_data or train_cfg.get("eval_data")
        if eval_data:
            eval_samples, eval_processed = materialize_training_data(
                str(eval_data),
                data_format=args.data_format,
                options=options,
                project_root=PROJECT_ROOT,
                source="local",
                split=args.data_split,
            )
            eval_records = [{"messages": messages, "text": render_text(messages, tokenizer)} for messages in eval_samples]
            LOGGER.info("Using dedicated eval file %s (%s samples)", eval_processed, len(eval_records))
        else:
            eval_split = args.eval_split
            if eval_split is None:
                eval_split = float(train_cfg.get("eval_split", 0.1))
            train_records, eval_records = split_records(records, float(eval_split), int(train_cfg.get("seed", 42)))
            if eval_records:
                records = train_records
                LOGGER.info(
                    "Split train/eval for eval_loss: train=%s eval=%s (eval_split=%s)",
                    len(records),
                    len(eval_records),
                    eval_split,
                )
            else:
                LOGGER.info("eval_split=%s produced no eval split; eval_loss is disabled.", eval_split)
        dataset = Dataset.from_list(records)
        eval_dataset = Dataset.from_list(eval_records) if eval_records else None
        model, tokenizer, peft_config = load_model_and_tokenizer(base_model, train_cfg)
        trainer = build_trainer(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            training_cfg=train_cfg,
            output_dir=output_dir,
            lora_cfg=peft_config,
            max_seq_length=int(train_cfg.get("max_seq_length", 2048)),
            eval_dataset=eval_dataset,
        )

        resume = args.resume_from_checkpoint
        if resume is None:
            resume = train_cfg.get("resume_from_checkpoint")
        if resume in {True, "true", "auto"}:
            found = latest_checkpoint(output_dir)
            resume = str(found) if found else None
            LOGGER.info("Auto resume checkpoint: %s", resume)
        elif resume in {False, "false", None, ""}:
            resume = None

        LOGGER.info("Starting training output_dir=%s resume=%s", output_dir, resume)
        report_to = str(train_cfg.get("report_to") or "none")
        if report_to not in {"", "none"}:
            LOGGER.info("Training report_to=%s (install extra `report` for tensorboard/wandb).", report_to)
        trainer.train(resume_from_checkpoint=resume)
        trainer.save_model(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))
        LOGGER.info("Saved adapter / trainer output to %s", output_dir)

        if bool(train_cfg.get("save_merged_model", False)):
            LOGGER.info("save_merged_model=true; merging adapter into a full model.")
            from merge_lora import main as merge_main

            sys.argv = [
                "merge_lora.py",
                "--base-model",
                str(base_model),
                "--adapter-path",
                str(output_dir),
                "--output-dir",
                str(output_dir / "merged"),
            ]
            merge_code = merge_main()
            if merge_code != 0:
                return merge_code

        rel_output = str(output_dir)
        try:
            rel_output = "./" + str(output_dir.resolve().relative_to(PROJECT_ROOT)).replace("\\", "/")
        except ValueError:
            pass
        LOGGER.info("Training finished. Update .env CHECKPOINT_PATH and ADAPTER_PATH to %s", rel_output)
        if args.update_env:
            upsert_env_key(config.env_path, "CHECKPOINT_PATH", rel_output)
            upsert_env_key(config.env_path, "ADAPTER_PATH", rel_output)
            upsert_env_key(config.env_path, "TRAINED_MODEL_MODE", "lora")
            LOGGER.info("Updated CHECKPOINT_PATH / ADAPTER_PATH in %s", config.env_path)
        print(output_dir)
        return 0
    except (ConfigError, DataFormatError) as exc:
        LOGGER.error("%s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
