#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from common.bootstrap import PROJECT_ROOT, ensure_sys_path

ensure_sys_path()

from common.data_format import DataOptions, parse_training_file
from common.env import ConfigError, load_app_config
from common.health import check_openai_models
from common.logging_utils import setup_logging

LOGGER = setup_logging("llm_tools.eval")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate processed data locally or against a running vLLM server.")
    parser.add_argument("--data", default=str(PROJECT_ROOT / "training" / "data" / "sample.txt"))
    parser.add_argument("--data-format", default="default")
    parser.add_argument("--remote", action="store_true", help="Query the running OpenAI-compatible server.")
    parser.add_argument("--max-samples", type=int, default=5)
    parser.add_argument("--model", default=None)
    return parser.parse_args()


def exact_match(pred: str, gold: str) -> float:
    return 1.0 if pred.strip() == gold.strip() else 0.0


def length_ratio(pred: str, gold: str) -> float:
    if not gold:
        return 0.0
    return min(len(pred), len(gold)) / max(len(pred), len(gold), 1)


def last_assistant(messages: list[dict[str, str]]) -> str:
    for item in reversed(messages):
        if item["role"] == "assistant":
            return item["content"]
    return ""


def prompt_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    if messages and messages[-1]["role"] == "assistant":
        return messages[:-1]
    return messages


def main() -> int:
    args = parse_args()
    try:
        data_path = Path(args.data)
        if not data_path.is_absolute():
            data_path = PROJECT_ROOT / data_path
        samples, report = parse_training_file(data_path, args.data_format, DataOptions())
        LOGGER.info("Parsed %s samples from %s", report.samples, data_path)
        metrics = {
            "samples": report.samples,
            "conversion": report.as_dict(),
            "exact_match": None,
            "length_ratio": None,
        }
        if not args.remote:
            LOGGER.info("Offline eval only checks conversion quality. Pass --remote to query vLLM.")
            print(json.dumps(metrics, ensure_ascii=False, indent=2))
            return 0

        config = load_app_config()
        host = config.require("VLLM_HOST")
        port = int(config.require("VLLM_PORT"))
        api_key = config.get("VLLM_API_KEY") or "sk-local"
        payload = check_openai_models(host, port, api_key=api_key)
        available = [item.get("id") for item in payload.get("data", [])]
        model_name = args.model or config.get("TRAINED_LORA_NAME") or config.get("VLLM_SERVED_MODEL_NAME")
        if not model_name and available:
            model_name = available[0]
        if not model_name:
            raise ConfigError("No model name available from /v1/models.")

        from openai import OpenAI

        check_host = "127.0.0.1" if host in {"0.0.0.0", "::", ""} else host
        client = OpenAI(base_url=f"http://{check_host}:{port}/v1", api_key=api_key)
        scores_em: list[float] = []
        scores_len: list[float] = []
        for sample in samples[: args.max_samples]:
            gold = last_assistant(sample)
            prompt = prompt_messages(sample)
            completion = client.chat.completions.create(model=model_name, messages=prompt, temperature=0)
            pred = completion.choices[0].message.content or ""
            scores_em.append(exact_match(pred, gold))
            scores_len.append(length_ratio(pred, gold))
            LOGGER.info("gold=%r pred=%r", gold[:120], pred[:120])
        metrics["exact_match"] = sum(scores_em) / len(scores_em) if scores_em else None
        metrics["length_ratio"] = sum(scores_len) / len(scores_len) if scores_len else None
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
        return 0
    except (ConfigError, Exception) as exc:
        LOGGER.error("%s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
