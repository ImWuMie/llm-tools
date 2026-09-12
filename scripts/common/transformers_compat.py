from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .logging_utils import setup_logging

LOGGER = setup_logging("llm_tools.transformers_compat")

_PATCHED_ATTR = "_llm_tools_tied_weights_list_compat"
_MASK_PATCHED_ATTR = "_llm_tools_mask_kwargs_compat"
_MASK_FUNCTIONS = (
    "create_causal_mask",
    "create_sliding_window_causal_mask",
    "create_chunked_causal_mask",
    "create_bidirectional_mask",
    "create_bidirectional_sliding_window_mask",
)


def list_tied_weights_to_dict(model: Any, keys: Sequence[str]) -> dict[str, str]:
    """Convert transformers 4.x list `_tied_weights_keys` to the 5.x dict mapping."""
    source = _input_embedding_weight_name(model)
    return {str(key): source for key in keys}


def _input_embedding_weight_name(model: Any) -> str:
    try:
        embed = model.get_input_embeddings()
        weight = getattr(embed, "weight", None)
        if weight is not None:
            for name, param in model.named_parameters(remove_duplicate=False):
                if param is weight:
                    return name
    except Exception:
        pass
    module_names = []
    try:
        module_names = [name for name, _ in model.named_modules()]
    except Exception:
        pass
    for candidate in (
        "model.embedding.weight",
        "model.embed_tokens.weight",
        "embed_tokens.weight",
        "transformer.wte.weight",
    ):
        module = candidate.removesuffix(".weight")
        if module in module_names or not module_names:
            return candidate
    return "model.embedding.weight"


def apply_tied_weights_list_compat() -> bool:
    """Make PreTrainedModel.get_expanded_tied_weights_keys accept list mappings."""
    try:
        from transformers.modeling_utils import PreTrainedModel
    except Exception:
        return False
    current = getattr(PreTrainedModel, "get_expanded_tied_weights_keys", None)
    if current is None or getattr(current, _PATCHED_ATTR, False):
        return False
    original = current

    def wrapper(self, all_submodels: bool = False):  # noqa: ANN001
        mapping = getattr(self, "_tied_weights_keys", None)
        if isinstance(mapping, (list, tuple)):
            converted = list_tied_weights_to_dict(self, [str(item) for item in mapping])
            try:
                type(self)._tied_weights_keys = converted
            except Exception:
                pass
            self._tied_weights_keys = converted
            LOGGER.info(
                "Converted _tied_weights_keys list %s to dict %s for transformers 5.x",
                list(mapping),
                converted,
            )
        return original(self, all_submodels=all_submodels)

    setattr(wrapper, _PATCHED_ATTR, True)
    PreTrainedModel.get_expanded_tied_weights_keys = wrapper  # type: ignore[method-assign]
    return True


_TIED_LIST_RE = re.compile(
    r"^([ \t]*_tied_weights_keys\s*=\s*)\[([^\]]*)\]([^\n]*)$",
    re.MULTILINE,
)


def patch_modeling_tied_weights_file(model_dir: Path) -> bool:
    """Rewrite list `_tied_weights_keys` in local custom modeling code if present."""
    changed_any = False
    for path in Path(model_dir).glob("modeling*.py"):
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        match = _TIED_LIST_RE.search(text)
        if match is None:
            continue
        inner = match.group(2)
        keys = [item.strip().strip("'\"") for item in inner.split(",") if item.strip().strip("'\"")]
        if not keys:
            continue
        source = "model.embedding.weight" if "self.embedding" in text else "model.embed_tokens.weight"
        mapping = ", ".join(f'"{key}": "{source}"' for key in keys)
        replacement = f"{match.group(1)}{{{mapping}}}{match.group(3)}"
        new_text = _TIED_LIST_RE.sub(replacement, text, count=1)
        if new_text == text:
            continue
        try:
            path.write_text(new_text, encoding="utf-8", newline="\n")
        except OSError as exc:
            LOGGER.warning("Cannot patch %s tied weight keys: %s", path, exc)
            continue
        LOGGER.info("Patched %s _tied_weights_keys list -> dict for transformers 5.x", path)
        changed_any = True
    return changed_any


def normalize_mask_kwargs(kwargs: dict[str, Any], accepted: set[str] | None = None) -> dict[str, Any]:
    """Adapt Spark/transformers-4 mask kwargs to transformers 5 signatures."""
    out = dict(kwargs)
    if "input_embeds" in out and "inputs_embeds" not in out:
        out["inputs_embeds"] = out.pop("input_embeds")
    elif "input_embeds" in out:
        out.pop("input_embeds")
    if accepted is None:
        return out
    if "inputs_embeds" in out and "inputs_embeds" not in accepted and "input_embeds" in accepted:
        out["input_embeds"] = out.pop("inputs_embeds")
    return {key: value for key, value in out.items() if key in accepted}


def _wrap_mask_function(fn):  # noqa: ANN001
    import inspect

    try:
        signature = inspect.signature(fn)
        accepted = set(signature.parameters)
        has_var_kw = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values())
    except (TypeError, ValueError):
        accepted = None
        has_var_kw = True

    def wrapper(*args, **kwargs):  # noqa: ANN002, ANN003
        filtered = normalize_mask_kwargs(kwargs, None if has_var_kw else accepted)
        return fn(*args, **filtered)

    setattr(wrapper, _MASK_PATCHED_ATTR, True)
    wrapper.__name__ = getattr(fn, "__name__", "wrapped_mask")
    wrapper.__wrapped__ = fn
    return wrapper


def apply_masking_kwargs_compat() -> bool:
    """Accept `input_embeds` and drop unknown kwargs such as `cache_position`."""
    try:
        import transformers.masking_utils as masking_utils
    except Exception:
        return False
    patched = False
    for name in _MASK_FUNCTIONS:
        fn = getattr(masking_utils, name, None)
        if fn is None or getattr(fn, _MASK_PATCHED_ATTR, False):
            continue
        setattr(masking_utils, name, _wrap_mask_function(fn))
        patched = True
    return patched


def patch_modeling_mask_kwargs_file(model_dir: Path) -> bool:
    """Rename mask kwarg `input_embeds` to `inputs_embeds` in local custom modeling code."""
    changed_any = False
    for path in Path(model_dir).glob("modeling*.py"):
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        new_text = text.replace('"input_embeds":', '"inputs_embeds":').replace("'input_embeds':", "'inputs_embeds':")
        if new_text == text:
            continue
        try:
            path.write_text(new_text, encoding="utf-8", newline="\n")
        except OSError as exc:
            LOGGER.warning("Cannot patch %s mask kwargs: %s", path, exc)
            continue
        LOGGER.info("Patched %s mask kwargs input_embeds -> inputs_embeds for transformers 5.x", path)
        changed_any = True
    return changed_any


def apply_transformers5_compat(model_dir: Path | None = None) -> None:
    if model_dir is not None:
        patch_modeling_tied_weights_file(model_dir)
        patch_modeling_mask_kwargs_file(model_dir)
    apply_tied_weights_list_compat()
    apply_masking_kwargs_compat()
