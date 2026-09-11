from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .data_format import DataFormatError, DataOptions, parse_training_file, write_processed_jsonl
from .logging_utils import setup_logging

LOGGER = setup_logging("llm_tools.hub_data")


def looks_like_hub_id(value: str) -> bool:
    text = value.strip()
    if not text or text.startswith(".") or "\\" in text:
        return False
    path = Path(text)
    if path.exists():
        return False
    return "/" in text and not text.startswith("/")


def _record_to_messages(record: dict[str, Any], options: DataOptions) -> list[dict[str, str]] | None:
    if isinstance(record.get("messages"), list):
        return [{"role": str(item.get("role")), "content": str(item.get("content", ""))} for item in record["messages"] if isinstance(item, dict)]
    if isinstance(record.get("conversations"), list):
        mapped = []
        for item in record["conversations"]:
            if not isinstance(item, dict):
                continue
            mapped.append({"role": str(item.get("from") or item.get("role")), "content": str(item.get("value") or item.get("content") or "")})
        return mapped
    instruction = str(record.get("instruction") or "").strip()
    user_input = str(record.get("input") or "").strip()
    output = str(record.get("output") or record.get("response") or "").strip()
    if instruction or user_input:
        user = instruction if not user_input else f"{instruction}\n{user_input}".strip()
        messages = [{"role": "user", "content": user}]
        if output:
            messages.append({"role": "assistant", "content": output})
        return messages
    user = str(record.get(options.custom_user_key) or "").strip()
    assistant = str(record.get(options.custom_assistant_key) or "").strip()
    if user and assistant:
        messages = [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]
        system = str(record.get(options.custom_system_key) or "").strip()
        if system:
            messages.insert(0, {"role": "system", "content": system})
        return messages
    return None


def records_to_samples(records: Iterable[dict[str, Any]], options: DataOptions) -> list[list[dict[str, str]]]:
    from .data_format import validate_messages, with_system

    samples: list[list[dict[str, str]]] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        messages = _record_to_messages(record, options)
        if not messages:
            continue
        try:
            samples.append(validate_messages(with_system(messages, options.system_prompt)))
        except DataFormatError:
            continue
    if not samples:
        raise DataFormatError("Hub dataset produced no valid chat samples.")
    return samples


def load_hf_dataset(dataset_id: str, *, split: str = "train", token: str | None = None) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise DataFormatError("datasets is missing. Run `uv sync --extra train` or `--extra download`.") from exc
    LOGGER.info("Loading Hugging Face dataset %s split=%s", dataset_id, split)
    dataset = load_dataset(dataset_id, split=split, token=token)
    return [dict(row) for row in dataset]


def load_modelscope_dataset(dataset_id: str, *, split: str = "train") -> list[dict[str, Any]]:
    try:
        from modelscope.msdatasets import MsDataset
    except Exception as exc:
        raise DataFormatError("modelscope is missing. Run `uv sync --extra download`.") from exc
    LOGGER.info("Loading ModelScope dataset %s split=%s", dataset_id, split)
    dataset = MsDataset.load(dataset_id, split=split)
    rows: list[dict[str, Any]] = []
    for row in dataset:
        rows.append(dict(row) if not isinstance(row, dict) else row)
    return rows


def materialize_training_data(
    spec: str,
    *,
    data_format: str,
    options: DataOptions,
    project_root: Path,
    source: str = "auto",
    split: str = "train",
    hf_token: str | None = None,
) -> tuple[list[list[dict[str, str]]], Path]:
    raw = spec.strip()
    local = Path(raw)
    if not local.is_absolute():
        local = (project_root / local).resolve()
    if local.exists():
        samples, _report = parse_training_file(local, data_format, options)
        processed = project_root / "training" / "data" / "processed" / f"{local.stem}.jsonl"
        write_processed_jsonl(samples, processed)
        return samples, processed

    if source == "local" or not looks_like_hub_id(raw):
        raise DataFormatError(f"Training data not found: {local}")

    chosen = source
    if chosen == "auto":
        chosen = "hf"
    if chosen == "hf":
        records = load_hf_dataset(raw, split=split, token=hf_token)
    elif chosen == "modelscope":
        records = load_modelscope_dataset(raw, split=split)
    else:
        raise DataFormatError(f"Unknown data source `{source}`.")

    samples = records_to_samples(records, options)
    raw_dir = project_root / "training" / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / f"{raw.replace('/', '__')}.jsonl"
    with raw_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    processed = project_root / "training" / "data" / "processed" / f"{raw_path.stem}.jsonl"
    write_processed_jsonl(samples, processed)
    LOGGER.info("Materialized hub dataset %s -> %s (%s samples)", raw, processed, len(samples))
    return samples, processed
