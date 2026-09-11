from .base import DownloadRequest, ModelSource, SourceError
from .hf_source import HuggingFaceSource
from .modelscope_source import ModelScopeSource

SOURCE_ALIASES = {
    "hf": "hf",
    "huggingface": "hf",
    "hugging_face": "hf",
    "modelscope": "modelscope",
    "ms": "modelscope",
}

SOURCE_FACTORY = {
    "hf": HuggingFaceSource,
    "modelscope": ModelScopeSource,
}


def normalize_source_name(name: str) -> str:
    key = name.strip().lower()
    if key not in SOURCE_ALIASES:
        raise SourceError(f"Unknown model source `{name}`. Use auto, hf, or modelscope.")
    return SOURCE_ALIASES[key]


def resolve_source_order(source: str, priority: str | None) -> list[str]:
    requested = source.strip().lower() or "auto"
    if requested == "auto":
        items = [item.strip() for item in (priority or "hf,modelscope").split(",") if item.strip()]
        if not items:
            items = ["hf", "modelscope"]
        return [normalize_source_name(item) for item in items]
    return [normalize_source_name(requested)]


def get_source(name: str) -> ModelSource:
    normalized = normalize_source_name(name)
    return SOURCE_FACTORY[normalized]()


__all__ = [
    "DownloadRequest",
    "ModelSource",
    "SourceError",
    "HuggingFaceSource",
    "ModelScopeSource",
    "resolve_source_order",
    "get_source",
]
