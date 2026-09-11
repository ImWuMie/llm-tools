from __future__ import annotations

import json
from pathlib import Path

TOKENIZER_CANDIDATES = (
    "tokenizer.json",
    "tokenizer.model",
    "vocab.json",
    "spiece.model",
    "tokenizer_config.json",
)
WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".gguf")


class ModelValidationError(RuntimeError):
    pass


def _nonempty_files(root: Path, names: tuple[str, ...]) -> list[Path]:
    found: list[Path] = []
    for name in names:
        path = root / name
        if path.is_file() and path.stat().st_size > 0:
            found.append(path)
    return found


def find_weight_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in WEIGHT_SUFFIXES and path.stat().st_size > 0:
            files.append(path)
    return files


def validate_local_model(model_dir: Path) -> list[str]:
    """Return a list of human-readable problems. Empty means valid."""
    problems: list[str] = []
    if not model_dir.exists():
        return [f"model directory does not exist: {model_dir}"]
    if not model_dir.is_dir():
        return [f"model path is not a directory: {model_dir}"]

    config = model_dir / "config.json"
    if not config.is_file() or config.stat().st_size == 0:
        problems.append("missing or empty config.json")

    tokenizer_hits = _nonempty_files(model_dir, TOKENIZER_CANDIDATES)
    vocab = model_dir / "vocab.json"
    merges = model_dir / "merges.txt"
    if not tokenizer_hits and not (vocab.is_file() and merges.is_file()):
        problems.append(
            "missing tokenizer files (expected tokenizer.json / tokenizer.model / vocab.json+merges.txt / tokenizer_config.json)"
        )

    weights = find_weight_files(model_dir)
    if not weights:
        problems.append("missing weight files (.safetensors / .bin / .pt / .gguf)")

    index = model_dir / "model.safetensors.index.json"
    if index.is_file():
        try:
            payload = json.loads(index.read_text(encoding="utf-8"))
            weight_map = payload.get("weight_map") or {}
            missing_shards = []
            for shard in sorted(set(weight_map.values())):
                shard_path = model_dir / shard
                if not shard_path.is_file() or shard_path.stat().st_size == 0:
                    missing_shards.append(shard)
            if missing_shards:
                problems.append("safetensors index references missing shards: " + ", ".join(missing_shards[:8]))
        except json.JSONDecodeError:
            problems.append("model.safetensors.index.json is not valid JSON")

    return problems


def is_valid_local_model(model_dir: Path) -> bool:
    return not validate_local_model(model_dir)


def looks_like_lora(path: Path) -> bool:
    if not path.exists():
        return False
    names = {
        "adapter_config.json",
        "adapter_model.safetensors",
        "adapter_model.bin",
        "adapter_model.pt",
    }
    if path.is_file():
        return path.name in names
    return (path / "adapter_config.json").is_file() or (path / "adapter_model.safetensors").is_file() or (
        path / "adapter_model.bin"
    ).is_file()


def looks_like_merged_model(path: Path) -> bool:
    return is_valid_local_model(path) and not looks_like_lora(path)
