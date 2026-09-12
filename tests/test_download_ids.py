from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from download_model import build_output_dir, model_id_for


class _Cfg:
    def __init__(self, values: dict[str, str]) -> None:
        self.values = values

    def get(self, key: str, default: str | None = None) -> str | None:
        value = self.values.get(key)
        if value is None or value == "":
            return default
        return value

    def get_path(self, key: str, default: str | None = None):
        value = self.get(key, default)
        return Path(value) if value else None


def _args(**kwargs) -> Namespace:
    base = dict(
        model_id=None,
        hf_model_id=None,
        modelscope_model_id=None,
        output_dir=None,
        output_name=None,
    )
    base.update(kwargs)
    return Namespace(**base)


def test_cli_model_id_overrides_env_source_ids() -> None:
    config = _Cfg(
        {
            "MODEL_ID": "Qwen/Qwen2.5-7B-Instruct",
            "HF_MODEL_ID": "Qwen/Qwen2.5-7B-Instruct",
            "MODELSCOPE_MODEL_ID": "Qwen/Qwen2.5-7B-Instruct",
        }
    )
    args = _args(model_id="Qwen/Qwen2.5-0.5B-Instruct")
    assert model_id_for("modelscope", args, config) == "Qwen/Qwen2.5-0.5B-Instruct"
    assert model_id_for("hf", args, config) == "Qwen/Qwen2.5-0.5B-Instruct"


def test_source_specific_cli_wins_over_generic_model_id() -> None:
    config = _Cfg({"MODELSCOPE_MODEL_ID": "env/ms", "HF_MODEL_ID": "env/hf", "MODEL_ID": "env/generic"})
    args = _args(model_id="cli/generic", modelscope_model_id="cli/ms", hf_model_id="cli/hf")
    assert model_id_for("modelscope", args, config) == "cli/ms"
    assert model_id_for("hf", args, config) == "cli/hf"


def test_cli_model_id_uses_its_own_folder(tmp_path: Path) -> None:
    config = _Cfg({"MODEL_OUTPUT_NAME": "Qwen2.5-7B-Instruct"})
    args = _args(model_id="Qwen/Qwen2.5-0.5B-Instruct", output_dir=str(tmp_path / "models"))
    out = build_output_dir(args, config)
    assert out.name == "Qwen2.5-0.5B-Instruct"
    assert out.parent == (tmp_path / "models").resolve()
