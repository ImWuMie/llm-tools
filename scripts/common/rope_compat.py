from __future__ import annotations

import builtins
import sys
from collections.abc import Callable, Iterable
from typing import Any

LAYER_ROPE_KEYS = {
    "full_attention",
    "sliding_attention",
    "linear_attention",
    "chunked_attention",
}

SPARK_ROPE_IGNORE_KEYS = {
    "partial_rotary_factor",
    "full_attention",
    "sliding_attention",
    "linear_attention",
    "chunked_attention",
}

_PATCHING = False
_HOOK_INSTALLED = False
_TRANSFORMERS_PATCHED = False
_SPARK_PATCHED = False


def is_per_layer_rope(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    if any(key in LAYER_ROPE_KEYS for key in value):
        return True
    nested = [item for item in value.values() if isinstance(item, dict)]
    if not nested:
        return False
    return "rope_type" not in value and "type" not in value


def _as_inner_rope(value: dict[str, Any], *, default_theta: Any = None, default_type: str = "default") -> dict[str, Any]:
    inner = dict(value)
    inner.setdefault("rope_type", inner.get("type", default_type) or default_type)
    if default_theta is not None and "rope_theta" not in inner:
        inner["rope_theta"] = default_theta
    return inner


def normalize_rope_parameters(value: Any) -> Any:
    """Normalize Spark / custom RoPE configs without injecting float siblings.

    Spark-X2.5 stores per-attention maps::

        {"full_attention": {"rope_theta": 5e6, "partial_rotary_factor": 0.25},
         "sliding_attention": {"rope_theta": 1e4, "partial_rotary_factor": 1.0}}

    Newer transformers ``convert_rope_params_to_dict`` then does
    ``rope_parameters.setdefault("rope_theta", 10000.0)``, leaving a float next to
    those nested dicts. ``validate_rope`` iterates *all* values and crashes with
    ``'float' object has no attribute 'get'``.
    """
    if isinstance(value, (int, float)):
        return {"rope_type": "default", "rope_theta": float(value)}
    if not isinstance(value, dict):
        return value
    if is_per_layer_rope(value):
        extra_theta = value.get("rope_theta") if isinstance(value.get("rope_theta"), (int, float)) else None
        extra_type = value.get("rope_type") or value.get("type") or "default"
        patched: dict[str, Any] = {}
        for key, item in value.items():
            if key in {"rope_theta", "rope_type", "type"} and not isinstance(item, dict):
                continue
            if isinstance(item, dict):
                patched[key] = _as_inner_rope(item, default_theta=extra_theta, default_type=str(extra_type))
            elif isinstance(item, (int, float)):
                patched[key] = _as_inner_rope({}, default_theta=item, default_type=str(extra_type))
            else:
                patched[key] = item
        return patched
    patched = dict(value)
    patched.setdefault("rope_type", patched.get("type", "default"))
    if "rope_theta" not in patched:
        for item in patched.values():
            if isinstance(item, dict) and "rope_theta" in item:
                patched["rope_theta"] = item["rope_theta"]
                break
    return patched


def apply_rope_to_config(config: Any) -> Any:
    current = getattr(config, "rope_parameters", None)
    if current is None:
        return current
    normalized = normalize_rope_parameters(current)
    if normalized is not current:
        try:
            config.rope_parameters = normalized
        except Exception:
            try:
                object.__setattr__(config, "rope_parameters", normalized)
            except Exception:
                pass
    ignore = set(getattr(config, "ignore_keys_at_rope_validation", None) or [])
    ignore.update(SPARK_ROPE_IGNORE_KEYS)
    try:
        config.ignore_keys_at_rope_validation = ignore
    except Exception:
        try:
            object.__setattr__(config, "ignore_keys_at_rope_validation", ignore)
        except Exception:
            pass
    return getattr(config, "rope_parameters", normalized)


def _log(message: str) -> None:
    sys.stderr.write(f"llm-tools: {message}\n")


def _fn_name(func: Any) -> str:
    target = getattr(func, "__func__", func)
    return str(getattr(target, "__name__", "") or "")


def replace_class_validators(cls: Any, original: Callable[..., Any], wrapped: Callable[..., Any]) -> int:
    """Replace a HuggingFace Hub ``@strict`` class validator in-place.

    ``huggingface_hub.dataclasses`` snapshots ``validate_*`` callables onto
    ``cls.__class_validators__`` at class-decoration time. Replacing the method
    on the class / module does **not** update that list.
    """
    validators = getattr(cls, "__class_validators__", None)
    if not isinstance(validators, list):
        return 0
    orig_target = getattr(original, "__func__", original)
    replaced = 0
    for index, func in enumerate(list(validators)):
        target = getattr(func, "__func__", func)
        if target is orig_target or _fn_name(func) == "validate_rope":
            if getattr(target, "_llm_tools_patched", False):
                continue
            validators[index] = wrapped
            replaced += 1
    return replaced


def _walk_subclasses(cls: Any) -> Iterable[Any]:
    try:
        subclasses = cls.__subclasses__()
    except Exception:
        return
    for sub in subclasses:
        yield sub
        yield from _walk_subclasses(sub)


def make_validate_rope_wrapper(original: Callable[..., Any]) -> Callable[..., Any]:
    def wrapped(config: Any, *args: Any, **kwargs: Any) -> Any:
        apply_rope_to_config(config)
        try:
            return original(config, *args, **kwargs)
        except AttributeError as exc:
            text = str(exc)
            if "has no attribute 'get'" in text or "has no attribute 'keys'" in text:
                return None
            raise
        except (TypeError, ValueError) as exc:
            text = str(exc).lower()
            if "rope" in text or "unexpected" in text or "partial_rotary" in text:
                return None
            raise

    wrapped._llm_tools_patched = True  # type: ignore[attr-defined]
    wrapped.__name__ = getattr(original, "__name__", "validate_rope")
    wrapped.__qualname__ = getattr(original, "__qualname__", wrapped.__name__)
    return wrapped


def make_convert_rope_wrapper(original: Callable[..., Any]) -> Callable[..., Any]:
    def wrapped(self: Any, **kwargs: Any) -> Any:
        current = getattr(self, "rope_parameters", None)
        if current is not None:
            apply_rope_to_config(self)
        result = original(self, **kwargs)
        apply_rope_to_config(self)
        return result

    wrapped._llm_tools_patched = True  # type: ignore[attr-defined]
    wrapped.__name__ = getattr(original, "__name__", "convert_rope_params_to_dict")
    wrapped.__qualname__ = getattr(original, "__qualname__", wrapped.__name__)
    return wrapped


def _patch_method(owner: Any, name: str, factory: Callable[[Callable[..., Any]], Callable[..., Any]]) -> Callable[..., Any] | None:
    original = getattr(owner, name, None)
    if original is None or not callable(original):
        return None
    if getattr(original, "_llm_tools_patched", False):
        return original
    wrapped = factory(original)
    try:
        setattr(owner, name, wrapped)
    except Exception:
        return None
    return wrapped


def patch_transformers_rope() -> bool:
    """Patch transformers RoPE conversion + the bound HF Hub validator."""
    global _TRANSFORMERS_PATCHED
    if _TRANSFORMERS_PATCHED:
        return True
    try:
        import transformers.configuration_utils as configuration_utils
        import transformers.modeling_rope_utils as rope_utils
    except Exception:
        return False

    mixin = getattr(rope_utils, "RotaryEmbeddingConfigMixin", None)
    owners: list[Any] = []
    if mixin is not None:
        owners.append(mixin)
    for attr in ("PreTrainedConfig", "PretrainedConfig"):
        cls = getattr(configuration_utils, attr, None)
        if cls is not None and cls not in owners:
            owners.append(cls)

    original_validate = None
    for owner in owners:
        candidate = getattr(owner, "validate_rope", None)
        if callable(candidate) and not getattr(candidate, "_llm_tools_patched", False):
            original_validate = candidate
            break
    if original_validate is None and mixin is not None:
        original_validate = getattr(mixin, "validate_rope", None)

    wrapped_validate = None
    if callable(original_validate):
        if getattr(original_validate, "_llm_tools_patched", False):
            wrapped_validate = original_validate
        else:
            wrapped_validate = make_validate_rope_wrapper(original_validate)
            for owner in owners:
                try:
                    owner.validate_rope = wrapped_validate
                except Exception:
                    continue

    if mixin is not None:
        _patch_method(mixin, "convert_rope_params_to_dict", make_convert_rope_wrapper)

    module_validate = getattr(rope_utils, "validate_rope", None)
    if callable(module_validate) and wrapped_validate is not None:
        rope_utils.validate_rope = wrapped_validate
    elif callable(module_validate) and not getattr(module_validate, "_llm_tools_patched", False):
        rope_utils.validate_rope = make_validate_rope_wrapper(module_validate)
        wrapped_validate = rope_utils.validate_rope

    replaced = 0
    if wrapped_validate is not None and original_validate is not None:
        seen: set[int] = set()
        for cls in [*owners, *([sub for owner in owners for sub in _walk_subclasses(owner)])]:
            ident = id(cls)
            if ident in seen:
                continue
            seen.add(ident)
            replaced += replace_class_validators(cls, original_validate, wrapped_validate)

    _TRANSFORMERS_PATCHED = True
    _log(f"patched transformers RoPE validator (class_validators={replaced})")
    return True


def patch_spark_plugin_config() -> bool:
    """Coerce Spark plugin config kwargs before PretrainedConfig validation."""
    global _SPARK_PATCHED
    if _SPARK_PATCHED:
        return True
    try:
        from vllm_spark2_5_plugin.spark2_5_config import Spark2_5Config
    except Exception:
        return False
    original = getattr(Spark2_5Config, "__init__", None)
    if original is None or getattr(original, "_llm_tools_patched", False):
        _SPARK_PATCHED = original is not None
        return _SPARK_PATCHED

    def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
        if "rope_parameters" in kwargs:
            kwargs["rope_parameters"] = normalize_rope_parameters(kwargs["rope_parameters"])
        return original(self, *args, **kwargs)

    wrapped._llm_tools_patched = True  # type: ignore[attr-defined]
    wrapped.__name__ = getattr(original, "__name__", "__init__")
    Spark2_5Config.__init__ = wrapped  # type: ignore[method-assign]
    _SPARK_PATCHED = True
    _log("patched vllm_spark2_5_plugin.Spark2_5Config")
    return True


def apply_all_patches(*, log: bool = False) -> dict[str, bool]:
    global _PATCHING
    if _PATCHING:
        return {"transformers": _TRANSFORMERS_PATCHED, "spark": _SPARK_PATCHED}
    _PATCHING = True
    try:
        install_import_hook()
        results = {
            "transformers": patch_transformers_rope(),
            "spark": patch_spark_plugin_config(),
        }
        if log:
            _log(f"runtime patches applied: {results}")
        return results
    finally:
        _PATCHING = False


def install_import_hook() -> None:
    """Re-apply patches when transformers / Spark plugin are imported later."""
    global _HOOK_INSTALLED
    if _HOOK_INSTALLED:
        return
    original_import = builtins.__import__
    watch = {
        "transformers",
        "transformers.modeling_rope_utils",
        "transformers.configuration_utils",
        "vllm_spark2_5_plugin",
        "vllm_spark2_5_plugin.spark2_5_config",
    }

    def hooked(name: str, globals: Any = None, locals: Any = None, fromlist: Any = (), level: int = 0) -> Any:
        module = original_import(name, globals, locals, fromlist, level)
        if name in watch or (fromlist and name in {"transformers", "vllm_spark2_5_plugin"}):
            if not _PATCHING:
                try:
                    apply_all_patches()
                except Exception as exc:
                    _log(f"import-hook RoPE patch failed: {type(exc).__name__}: {exc}")
        return module

    builtins.__import__ = hooked  # type: ignore[assignment]
    _HOOK_INSTALLED = True
