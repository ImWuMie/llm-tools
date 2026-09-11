from __future__ import annotations

import json
from pathlib import Path

from common.validate_model import is_valid_local_model, looks_like_merged_model, validate_local_model


def _write_fake_model(root: Path) -> None:
    (root / "config.json").write_text('{"model_type":"qwen2"}', encoding="utf-8")
    (root / "tokenizer.json").write_text("{}", encoding="utf-8")
    (root / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (root / "model.safetensors").write_bytes(b"fake-weight")


def test_valid_model(tmp_path: Path) -> None:
    model = tmp_path / "m"
    model.mkdir()
    _write_fake_model(model)
    assert validate_local_model(model) == []
    assert is_valid_local_model(model)
    assert looks_like_merged_model(model)


def test_index_missing_shard(tmp_path: Path) -> None:
    model = tmp_path / "m"
    model.mkdir()
    _write_fake_model(model)
    (model / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a": "model-00001-of-00001.safetensors"}}),
        encoding="utf-8",
    )
    problems = validate_local_model(model)
    assert any("missing shards" in item for item in problems)
