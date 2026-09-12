from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .logging_utils import setup_logging

LOGGER = setup_logging("llm_tools.model_config")


def is_per_layer_rope(rope: Any) -> bool:
    return isinstance(rope, dict) and any(isinstance(value, dict) for value in rope.values())


def sanitize_rope_parameters(rope: Any) -> tuple[Any, bool]:
    """Make nested Spark-style rope_parameters safe for transformers 5.x.

    ``convert_rope_params_to_dict`` always does ``setdefault("rope_theta", 10000.0)``.
    That injects a float sibling next to ``full_attention`` / ``sliding_attention``,
    and ``validate_rope`` then crashes with ``'float' object has no attribute 'get'``.

    A dict sentinel at ``rope_theta`` makes setdefault a no-op and stays iterable.
    Nested per-layer maps are left intact for Spark / the vLLM plugin.
    """
    if not is_per_layer_rope(rope):
        return rope, False
    changed = False
    out: dict[str, Any] = {}
    for key, value in rope.items():
        if isinstance(value, dict):
            nested = dict(value)
            if "rope_type" not in nested:
                nested["rope_type"] = nested.get("type", "default")
                changed = True
            out[key] = nested
            continue
        changed = True
    if not isinstance(out.get("rope_theta"), dict):
        out["rope_theta"] = {"rope_type": "default"}
        changed = True
    return out, changed


def ensure_compatible_model_config(model_dir: Path, *, write: bool = True) -> bool:
    """Rewrite config.json in place when nested RoPE would crash transformers 5.x."""
    path = Path(model_dir) / "config.json"
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        LOGGER.warning("Cannot read %s: %s", path, exc)
        return False
    if not isinstance(payload, dict):
        return False
    new_rope, changed = sanitize_rope_parameters(payload.get("rope_parameters"))
    if not changed:
        return False
    if not write:
        return True
    payload["rope_parameters"] = new_rope
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    try:
        path.write_text(text, encoding="utf-8", newline="\n")
    except OSError as exc:
        LOGGER.warning("Cannot update %s for transformers 5.x RoPE compatibility: %s", path, exc)
        return False
    LOGGER.info(
        "Updated %s nested rope_parameters for transformers 5.x (dict sentinel at rope_theta).",
        path,
    )
    return True
