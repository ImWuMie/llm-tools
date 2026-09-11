from __future__ import annotations

import pytest

from model_sources import SourceError, resolve_source_order


def test_auto_priority() -> None:
    assert resolve_source_order("auto", "modelscope,hf") == ["modelscope", "hf"]
    assert resolve_source_order("hf", "modelscope,hf") == ["hf"]
    assert resolve_source_order("modelscope", None) == ["modelscope"]


def test_unknown_source() -> None:
    with pytest.raises(SourceError):
        resolve_source_order("gitee", None)
