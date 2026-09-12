from __future__ import annotations

from typing import Any


def normalize_rope_parameters(value: Any) -> Any:
    """Make Spark / custom rope configs look like transformers' expected dict."""
    if isinstance(value, (int, float)):
        return {"rope_type": "default", "rope_theta": float(value)}
    if not isinstance(value, dict):
        return value
    if "rope_type" in value or "type" in value:
        return value
    patched = dict(value)
    patched.setdefault("rope_type", "default")
    if "rope_theta" not in patched:
        for item in patched.values():
            if isinstance(item, dict) and "rope_theta" in item:
                patched["rope_theta"] = item["rope_theta"]
                break
    return patched


def patch_transformers_rope() -> bool:
    """Patch transformers.modeling_rope_utils.validate_rope for Spark configs.

    Newer transformers expect ``rope_parameters`` to be a dict with ``rope_type``.
    Spark2.5 stores nested per-attention maps or a bare theta float.
    """
    try:
        import transformers.modeling_rope_utils as rope_utils
    except Exception:
        return False
    original = getattr(rope_utils, "validate_rope", None)
    if original is None or getattr(original, "_llm_tools_patched", False):
        return original is not None

    def wrapped(config, *args, **kwargs):  # type: ignore[no-untyped-def]
        current = getattr(config, "rope_parameters", None)
        normalized = normalize_rope_parameters(current)
        if normalized is not current:
            try:
                config.rope_parameters = normalized
            except Exception:
                try:
                    object.__setattr__(config, "rope_parameters", normalized)
                except Exception:
                    pass
        return original(config, *args, **kwargs)

    wrapped._llm_tools_patched = True  # type: ignore[attr-defined]
    rope_utils.validate_rope = wrapped
    return True
