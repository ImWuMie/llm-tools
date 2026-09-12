from __future__ import annotations

from types import SimpleNamespace

from common.hf_server import (
    build_generate_kwargs,
    from_pretrained_kwargs,
    openai_stream_chunk,
    resolve_enable_thinking,
    resolve_max_tokens,
    resolve_temperature,
)


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


def test_resolve_max_tokens_prefers_completion_field() -> None:
    assert resolve_max_tokens({}) == 256
    assert resolve_max_tokens({"max_tokens": 32}) == 32
    assert resolve_max_tokens({"max_completion_tokens": 64}) == 64
    assert resolve_max_tokens({"max_tokens": 0}) == 1


def test_resolve_temperature_keeps_zero() -> None:
    assert resolve_temperature({}) == 0.0
    assert resolve_temperature({"temperature": 0}) == 0.0
    assert resolve_temperature({"temperature": 0.7}) == 0.7


def test_build_generate_kwargs_omits_temperature_when_greedy() -> None:
    tokenizer = SimpleNamespace(pad_token_id=2, eos_token_id=1)
    greedy = build_generate_kwargs(max_new_tokens=16, temperature=0, tokenizer=tokenizer)
    assert greedy["do_sample"] is False
    assert "temperature" not in greedy
    sampled = build_generate_kwargs(max_new_tokens=16, temperature=0.8, tokenizer=tokenizer)
    assert sampled["do_sample"] is True
    assert sampled["temperature"] == 0.8


def test_openai_stream_chunk_matches_vllm_chat_shape() -> None:
    first = openai_stream_chunk(
        completion_id="chatcmpl-x",
        created=1,
        model_name="spark",
        delta={"role": "assistant"},
    )
    assert first["object"] == "chat.completion.chunk"
    assert first["choices"][0]["delta"] == {"role": "assistant"}
    assert first["choices"][0]["finish_reason"] is None
    last = openai_stream_chunk(
        completion_id="chatcmpl-x",
        created=1,
        model_name="spark",
        delta={},
        finish_reason="stop",
        usage={"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
    )
    assert last["choices"][0]["finish_reason"] == "stop"
    assert last["usage"]["completion_tokens"] == 2


def test_resolve_enable_thinking(monkeypatch) -> None:
    monkeypatch.delenv("ENABLE_THINKING", raising=False)
    monkeypatch.delenv("SPARK_ENABLE_THINKING", raising=False)
    assert resolve_enable_thinking({}) is False
    assert resolve_enable_thinking({"enable_thinking": True}) is True
    assert resolve_enable_thinking({"chat_template_kwargs": {"enable_thinking": True}}) is True
    monkeypatch.setenv("ENABLE_THINKING", "1")
    assert resolve_enable_thinking({}) is True
