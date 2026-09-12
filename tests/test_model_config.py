from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from common.model_config import ensure_compatible_model_config, sanitize_rope_parameters


NESTED = {
    "full_attention": {"partial_rotary_factor": 0.25, "rope_theta": 5000000},
    "sliding_attention": {"partial_rotary_factor": 1.0, "rope_theta": 10000},
}

LAYER_TYPES = ["sliding_attention", "full_attention"]


def emulate_transformers5_validate(rope: dict, layer_types: list[str]) -> None:
    rope_parameters = copy.deepcopy(rope)
    rope_parameters.setdefault("rope_theta", 10000.0)
    if layer_types is not None and not set(rope_parameters.keys()).isdisjoint(layer_types):
        rope_parameters_dict = rope_parameters
    else:
        rope_parameters_dict = {"full_attention": rope_parameters}
    for item in rope_parameters_dict.values():
        if item is None:
            continue
        item.get("rope_type", item.get("type", "default"))


def test_unsanitized_nested_rope_crashes_like_transformers5() -> None:
    with pytest.raises(AttributeError):
        emulate_transformers5_validate(NESTED, LAYER_TYPES)


def test_sanitized_nested_rope_survives_transformers5_validate() -> None:
    out, changed = sanitize_rope_parameters(NESTED)
    assert changed is True
    assert out["full_attention"]["rope_theta"] == 5000000
    assert out["sliding_attention"]["partial_rotary_factor"] == 1.0
    assert out["rope_theta"] == {"rope_type": "default"}
    emulate_transformers5_validate(out, LAYER_TYPES)
    again, changed_again = sanitize_rope_parameters(out)
    assert changed_again is False
    assert again["rope_theta"] == {"rope_type": "default"}


def test_float_sibling_is_replaced_with_dict_sentinel() -> None:
    injected = dict(NESTED)
    injected["rope_theta"] = 10000.0
    out, changed = sanitize_rope_parameters(injected)
    assert changed is True
    assert out["rope_theta"] == {"rope_type": "default"}
    emulate_transformers5_validate(out, LAYER_TYPES)


def test_ensure_compatible_model_config_writes_once(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps({"model_type": "spark2_5", "rope_parameters": NESTED, "layer_types": LAYER_TYPES}),
        encoding="utf-8",
    )
    assert ensure_compatible_model_config(tmp_path) is True
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["rope_parameters"]["rope_theta"] == {"rope_type": "default"}
    assert payload["rope_parameters"]["full_attention"]["rope_type"] == "default"
    assert ensure_compatible_model_config(tmp_path) is False
