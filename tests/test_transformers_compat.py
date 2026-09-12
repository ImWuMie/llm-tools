from __future__ import annotations

from types import SimpleNamespace

from common.transformers_compat import list_tied_weights_to_dict, patch_modeling_tied_weights_file
from common.transformers_compat import (
    _wrap_mask_function,
    list_tied_weights_to_dict,
    normalize_mask_kwargs,
    patch_modeling_mask_kwargs_file,
    patch_modeling_tied_weights_file,
)


class _Embed(SimpleNamespace):
    pass


class _Model:
    def __init__(self) -> None:
        self._weight = object()
        self.embed = _Embed(weight=self._weight)

    def get_input_embeddings(self):
        return self.embed

    def named_parameters(self, remove_duplicate: bool = False):
        return [("model.embedding.weight", self._weight)]


def test_list_tied_weights_to_dict_uses_input_embeddings() -> None:
    mapping = list_tied_weights_to_dict(_Model(), ["lm_head.weight"])
    assert mapping == {"lm_head.weight": "model.embedding.weight"}


def test_patch_modeling_tied_weights_file(tmp_path) -> None:
    path = tmp_path / "modeling_spark.py"
    path.write_text(
        "class Spark2_5ForCausalLM:\n"
        '    _tied_weights_keys = ["lm_head.weight"]  # noqa: RUF012\n'
        "    def __init__(self):\n"
        "        self.embedding = None\n",
        encoding="utf-8",
    )
    assert patch_modeling_tied_weights_file(tmp_path) is True
    text = path.read_text(encoding="utf-8")
    assert '{"lm_head.weight": "model.embedding.weight"}' in text
    assert patch_modeling_tied_weights_file(tmp_path) is False


def test_normalize_mask_kwargs_renames_and_drops() -> None:
    out = normalize_mask_kwargs(
        {
            "config": "cfg",
            "input_embeds": "emb",
            "attention_mask": None,
            "cache_position": 1,
            "past_key_values": None,
            "position_ids": None,
        },
        {"config", "inputs_embeds", "attention_mask", "past_key_values", "position_ids"},
    )
    assert out["inputs_embeds"] == "emb"
    assert "input_embeds" not in out
    assert "cache_position" not in out
    assert set(out) == {"config", "inputs_embeds", "attention_mask", "past_key_values", "position_ids"}


def test_wrap_mask_function_accepts_spark_kwargs() -> None:
    seen: dict[str, object] = {}

    def fake_create_causal_mask(config, inputs_embeds, attention_mask, past_key_values, position_ids=None):
        seen.update(
            {
                "config": config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
        )
        return "ok"

    wrapped = _wrap_mask_function(fake_create_causal_mask)
    result = wrapped(
        config="cfg",
        input_embeds="emb",
        attention_mask=None,
        cache_position=7,
        past_key_values=None,
        position_ids=None,
    )
    assert result == "ok"
    assert seen["inputs_embeds"] == "emb"
    assert "cache_position" not in seen


def test_patch_modeling_mask_kwargs_file(tmp_path) -> None:
    path = tmp_path / "modeling_spark.py"
    path.write_text(
        'mask_kwargs = {"config": self.config, "input_embeds": inputs_embeds, "cache_position": cache_position}\n',
        encoding="utf-8",
    )
    assert patch_modeling_mask_kwargs_file(tmp_path) is True
    text = path.read_text(encoding="utf-8")
    assert '"inputs_embeds": inputs_embeds' in text
    assert '"input_embeds":' not in text
    assert patch_modeling_mask_kwargs_file(tmp_path) is False
