from __future__ import annotations

from common.hf_server import from_pretrained_kwargs


def test_from_pretrained_kwargs_skips_device_map_without_accelerate() -> None:
    kwargs = from_pretrained_kwargs(cuda=True, dtype="bf16", has_accelerate=False)
    assert "device_map" not in kwargs
    assert kwargs["trust_remote_code"] is True
    assert kwargs["dtype"] == "bf16"


def test_from_pretrained_kwargs_uses_device_map_with_accelerate() -> None:
    kwargs = from_pretrained_kwargs(cuda=True, dtype="bf16", has_accelerate=True)
    assert kwargs["device_map"] == "auto"


def test_from_pretrained_kwargs_cpu_has_no_device_map() -> None:
    kwargs = from_pretrained_kwargs(cuda=False, dtype="fp32", has_accelerate=True)
    assert "device_map" not in kwargs
