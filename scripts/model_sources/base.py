from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
import inspect
from typing import Any, Callable


class SourceError(RuntimeError):
    """Raised when a model source cannot complete a download."""


@dataclass(frozen=True)
class DownloadRequest:
    source: str
    model_id: str
    revision: str
    output_dir: Path
    token: str | None = None
    endpoint: str | None = None
    extra: dict | None = None


class ModelSource(ABC):
    name: str

    @abstractmethod
    def download(self, request: DownloadRequest) -> Path:
        raise NotImplementedError


def call_supported(fn: Callable[..., Any], **kwargs: Any) -> Any:
    params = inspect.signature(fn).parameters
    filtered = {key: value for key, value in kwargs.items() if key in params and value is not None}
    return fn(**filtered)
