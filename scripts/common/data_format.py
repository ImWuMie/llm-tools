from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

ROLE_ALIASES = {
    "user": "user",
    "human": "user",
    "instruction": "user",
    "prompt": "user",
    "assistant": "assistant",
    "gpt": "assistant",
    "bot": "assistant",
    "chatgpt": "assistant",
    "model": "assistant",
    "system": "system",
}


class DataFormatError(ValueError):
    """Raised when a training file cannot be parsed."""


@dataclass
class DataOptions:
    skip_empty_lines: bool = True
    skip_comment_lines: bool = True
    comment_prefix: str = "#"
    encoding: str = "utf-8"
    system_prompt: str = ""
    custom_user_key: str = "user"
    custom_assistant_key: str = "assistant"
    custom_system_key: str = "system"


@dataclass
class ConversionReport:
    source_path: str
    format_name: str
    raw_lines: int = 0
    skipped_empty: int = 0
    skipped_comments: int = 0
    dropped_unpaired: int = 0
    utf8_ok: bool = True
    samples: int = 0
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_path": self.source_path,
            "format": self.format_name,
            "raw_lines": self.raw_lines,
            "skipped_empty": self.skipped_empty,
            "skipped_comments": self.skipped_comments,
            "dropped_unpaired": self.dropped_unpaired,
            "utf8_ok": self.utf8_ok,
            "samples": self.samples,
            "warnings": self.warnings,
        }


def load_system_prompt(path: Path | None) -> str:
    if path is None or not path.exists():
        return ""
    text = path.read_text(encoding="utf-8").strip()
    return text


def normalize_role(role: str) -> str:
    key = (role or "").strip().lower()
    if key not in ROLE_ALIASES:
        raise DataFormatError(f"Unsupported role: {role!r}")
    return ROLE_ALIASES[key]


def with_system(messages: list[dict[str, str]], system_prompt: str) -> list[dict[str, str]]:
    cleaned = [dict(item) for item in messages]
    if not system_prompt.strip():
        return cleaned
    if cleaned and cleaned[0].get("role") == "system":
        if not cleaned[0].get("content"):
            cleaned[0]["content"] = system_prompt
        return cleaned
    return [{"role": "system", "content": system_prompt}, *cleaned]


def validate_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    if not messages:
        raise DataFormatError("empty messages")
    normalized: list[dict[str, str]] = []
    for item in messages:
        role = normalize_role(str(item.get("role", "")))
        content = str(item.get("content", "")).strip()
        if not content:
            continue
        normalized.append({"role": role, "content": content})
    if not any(msg["role"] == "user" for msg in normalized):
        raise DataFormatError("sample has no user message")
    if not any(msg["role"] == "assistant" for msg in normalized):
        raise DataFormatError("sample has no assistant message")
    return normalized


def _read_text(path: Path, options: DataOptions, report: ConversionReport) -> str:
    try:
        text = path.read_text(encoding=options.encoding)
        report.utf8_ok = True
        return text
    except UnicodeDecodeError as exc:
        report.utf8_ok = False
        raise DataFormatError(f"{path} is not valid {options.encoding}: {exc}") from exc


def _iter_content_lines(text: str, options: DataOptions, report: ConversionReport) -> Iterator[str]:
    for raw in text.splitlines():
        report.raw_lines += 1
        line = raw.strip()
        if options.skip_empty_lines and not line:
            report.skipped_empty += 1
            continue
        if options.skip_comment_lines and line.startswith(options.comment_prefix):
            report.skipped_comments += 1
            continue
        yield raw.rstrip("\n")


def parse_default_txt(path: Path, options: DataOptions) -> tuple[list[list[dict[str, str]]], ConversionReport]:
    report = ConversionReport(source_path=str(path), format_name="default")
    text = _read_text(path, options, report)
    lines = list(_iter_content_lines(text, options, report))
    if len(lines) % 2 != 0:
        report.dropped_unpaired = 1
        report.warnings.append("Odd number of content lines; dropping the last unpaired user line.")
        lines = lines[:-1]
    samples: list[list[dict[str, str]]] = []
    for index in range(0, len(lines), 2):
        user = lines[index].strip()
        assistant = lines[index + 1].strip()
        if not user or not assistant:
            report.warnings.append(f"Skipping empty pair starting at content line {index + 1}.")
            continue
        messages = with_system(
            [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}],
            options.system_prompt,
        )
        samples.append(validate_messages(messages))
    report.samples = len(samples)
    return samples, report


def _load_json_records(path: Path, options: DataOptions, report: ConversionReport) -> list[Any]:
    text = _read_text(path, options, report)
    stripped = text.strip()
    if not stripped:
        return []
    if path.suffix.lower() == ".json" and stripped[0] in "[{":
        payload = json.loads(stripped)
        return payload if isinstance(payload, list) else [payload]
    records: list[Any] = []
    for raw in text.splitlines():
        report.raw_lines += 1
        line = raw.strip()
        if options.skip_empty_lines and not line:
            report.skipped_empty += 1
            continue
        if options.skip_comment_lines and line.startswith(options.comment_prefix):
            report.skipped_comments += 1
            continue
        records.append(json.loads(line))
    if records:
        report.raw_lines = max(report.raw_lines, len(text.splitlines()))
    return records


def parse_jsonl_messages(path: Path, options: DataOptions) -> tuple[list[list[dict[str, str]]], ConversionReport]:
    report = ConversionReport(source_path=str(path), format_name="jsonl")
    records = _load_json_records(path, options, report)
    samples: list[list[dict[str, str]]] = []
    for record in records:
        if isinstance(record, dict) and "messages" in record:
            raw_messages = record["messages"]
        elif isinstance(record, list):
            raw_messages = record
        else:
            report.warnings.append("Skipping JSONL row without messages.")
            continue
        samples.append(validate_messages(with_system(raw_messages, options.system_prompt)))
    report.samples = len(samples)
    return samples, report


def parse_sharegpt(path: Path, options: DataOptions) -> tuple[list[list[dict[str, str]]], ConversionReport]:
    report = ConversionReport(source_path=str(path), format_name="sharegpt")
    records = _load_json_records(path, options, report)
    samples: list[list[dict[str, str]]] = []
    for record in records:
        conversations = record.get("conversations") if isinstance(record, dict) else None
        if not conversations:
            report.warnings.append("Skipping ShareGPT row without conversations.")
            continue
        messages: list[dict[str, str]] = []
        for turn in conversations:
            role = normalize_role(str(turn.get("from") or turn.get("role") or ""))
            content = str(turn.get("value") or turn.get("content") or "").strip()
            if content:
                messages.append({"role": role, "content": content})
        samples.append(validate_messages(with_system(messages, options.system_prompt)))
    report.samples = len(samples)
    return samples, report


def parse_alpaca(path: Path, options: DataOptions) -> tuple[list[list[dict[str, str]]], ConversionReport]:
    report = ConversionReport(source_path=str(path), format_name="alpaca")
    records = _load_json_records(path, options, report)
    samples: list[list[dict[str, str]]] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        instruction = str(record.get("instruction") or "").strip()
        user_input = str(record.get("input") or "").strip()
        output = str(record.get("output") or "").strip()
        user = instruction if not user_input else f"{instruction}\n{user_input}"
        messages = [{"role": "user", "content": user}, {"role": "assistant", "content": output}]
        samples.append(validate_messages(with_system(messages, options.system_prompt)))
    report.samples = len(samples)
    return samples, report


def parse_custom(path: Path, options: DataOptions) -> tuple[list[list[dict[str, str]]], ConversionReport]:
    report = ConversionReport(source_path=str(path), format_name="custom")
    records = _load_json_records(path, options, report)
    samples: list[list[dict[str, str]]] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        messages: list[dict[str, str]] = []
        system = str(record.get(options.custom_system_key) or "").strip()
        user = str(record.get(options.custom_user_key) or "").strip()
        assistant = str(record.get(options.custom_assistant_key) or "").strip()
        if system:
            messages.append({"role": "system", "content": system})
        messages.extend(
            [
                {"role": "user", "content": user},
                {"role": "assistant", "content": assistant},
            ]
        )
        samples.append(validate_messages(with_system(messages, options.system_prompt)))
    report.samples = len(samples)
    return samples, report


PARSERS = {
    "default": parse_default_txt,
    "txt": parse_default_txt,
    "jsonl": parse_jsonl_messages,
    "sharegpt": parse_sharegpt,
    "alpaca": parse_alpaca,
    "custom": parse_custom,
}


def parse_training_file(
    path: Path,
    data_format: str,
    options: DataOptions | None = None,
) -> tuple[list[list[dict[str, str]]], ConversionReport]:
    fmt = data_format.strip().lower()
    if fmt not in PARSERS:
        raise DataFormatError(f"Unsupported data format `{data_format}`. Choose from: {', '.join(PARSERS)}")
    parser = PARSERS[fmt]
    samples, report = parser(path, options or DataOptions())
    if not samples:
        raise DataFormatError(f"No valid samples parsed from {path} using format `{fmt}`.")
    return samples, report


def write_processed_jsonl(samples: Iterable[list[dict[str, str]]], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for messages in samples:
            handle.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
            count += 1
    if count == 0:
        raise DataFormatError("Refusing to write empty processed dataset.")
    return output_path
