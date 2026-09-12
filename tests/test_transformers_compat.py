from __future__ import annotations

from types import SimpleNamespace

from common.transformers_compat import list_tied_weights_to_dict, patch_modeling_tied_weights_file


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
