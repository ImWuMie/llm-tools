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
from common.metrics import assistant_turns, bleu, exact_match, length_ratio, rouge_l, token_f1
from datetime import datetime, timezone

LOGGER = setup_logging("llm_tools.eval")


def _write_eval(metrics: dict, output: str | None, default_dir: Path) -> Path:
    default_dir.mkdir(parents=True, exist_ok=True)
    path = Path(output) if output else default_dir / f"eval-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("Wrote eval report to %s", path)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate processed data locally or against a running vLLM server.")
    parser.add_argument("--data", default=str(PROJECT_ROOT / "training" / "data" / "sample.txt"))
    parser.add_argument("--data-format", default="default")
    parser.add_argument("--remote", action="store_true", help="Query the running OpenAI-compatible server.")
    parser.add_argument("--max-samples", type=int, default=5)
    parser.add_argument("--model", default=None)
    parser.add_argument("--output", default=None, help="Write JSON metrics to this path. Default logs/eval/.")
    return parser.parse_args()


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
            "token_f1": None,
            "bleu": None,
            "rouge_l": None,
            "turns": 0,
        }
        if not args.remote:
            LOGGER.info("Offline eval only checks conversion quality. Pass --remote to query the server.")
            _write_eval(metrics, args.output, PROJECT_ROOT / "logs" / "eval")
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
        scores_f1: list[float] = []
        scores_bleu: list[float] = []
        scores_rouge: list[float] = []
        turns = 0
        for sample in samples[: args.max_samples]:
            for prompt, gold in assistant_turns(sample):
                completion = client.chat.completions.create(model=model_name, messages=prompt, temperature=0)
                pred = completion.choices[0].message.content or ""
                scores_em.append(exact_match(pred, gold))
                scores_len.append(length_ratio(pred, gold))
                scores_f1.append(token_f1(pred, gold))
                scores_bleu.append(bleu(pred, gold))
                scores_rouge.append(rouge_l(pred, gold))
                turns += 1
                LOGGER.info("gold=%r pred=%r", gold[:120], pred[:120])
        def avg(values: list[float]) -> float | None:
            return sum(values) / len(values) if values else None

        metrics["turns"] = turns
        metrics["exact_match"] = avg(scores_em)
        metrics["length_ratio"] = avg(scores_len)
        metrics["token_f1"] = avg(scores_f1)
        metrics["bleu"] = avg(scores_bleu)
        metrics["rouge_l"] = avg(scores_rouge)
        _write_eval(metrics, args.output, PROJECT_ROOT / "logs" / "eval")
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
        return 0
    except (ConfigError, Exception) as exc:
        LOGGER.error("%s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
