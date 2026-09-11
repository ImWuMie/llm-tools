from __future__ import annotations

from pathlib import Path

from common.bootstrap import PROJECT_ROOT
from common.paths import resolve_path, to_wsl_path
from common.validate_model import is_valid_local_model, looks_like_lora, validate_local_model


def test_resolve_relative_path() -> None:
    path = resolve_path("./training/output", PROJECT_ROOT)
    assert path == (PROJECT_ROOT / "training" / "output").resolve()


def test_to_wsl_path_windows_style() -> None:
    fake = Path("D:/jetb/PycharmProjects/llm-tools")
    converted = to_wsl_path(fake)
    assert converted.startswith("/mnt/") or "/" in converted


def test_validate_incomplete_model(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    problems = validate_local_model(model_dir)
    assert problems
    assert is_valid_local_model(model_dir) is False


def test_looks_like_lora(tmp_path: Path) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    assert looks_like_lora(adapter)
