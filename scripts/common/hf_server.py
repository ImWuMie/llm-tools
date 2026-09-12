from __future__ import annotations

import json
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .logging_utils import setup_logging
from .model_config import ensure_compatible_model_config
from .transformers_compat import apply_transformers5_compat

LOGGER = setup_logging("llm_tools.hf_server")

_STATE: dict[str, Any] = {}


def _authorize(handler: BaseHTTPRequestHandler, api_key: str | None) -> bool:
    if not api_key:
        return True
    header = handler.headers.get("Authorization") or ""
    return header.strip() == f"Bearer {api_key}"


def _json(handler: BaseHTTPRequestHandler, code: int, payload: dict[str, Any]) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def accelerate_available() -> bool:
    try:
        import accelerate  # noqa: F401
    except Exception:
        return False
    return True


def from_pretrained_kwargs(*, cuda: bool, dtype: Any, has_accelerate: bool | None = None) -> dict[str, Any]:
    """Build AutoModelForCausalLM.from_pretrained kwargs.

    ``device_map='auto'`` requires accelerate. Single-GPU serving works without it.
    """
    kwargs: dict[str, Any] = {"trust_remote_code": True, "dtype": dtype}
    if cuda and (accelerate_available() if has_accelerate is None else has_accelerate):
        kwargs["device_map"] = "auto"
    return kwargs


def _model_device(model) -> Any:
    device = getattr(model, "device", None)
    if device is not None and str(device) != "meta":
        return device
    try:
        return next(model.parameters()).device
    except StopIteration:
        import torch

        return torch.device("cpu")


def load_causal_lm(model_dir: Path, adapter_path: Path | None = None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ensure_compatible_model_config(model_dir)
    apply_transformers5_compat(model_dir)
    LOGGER.info("Loading transformers model from %s", model_dir)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    cuda = bool(torch.cuda.is_available())
    dtype = torch.bfloat16 if cuda else torch.float32
    has_acc = accelerate_available()
    kwargs = from_pretrained_kwargs(cuda=cuda, dtype=dtype, has_accelerate=has_acc)
    if cuda and "device_map" not in kwargs:
        LOGGER.warning(
            "accelerate is not installed; loading onto a single CUDA device without device_map. "
            "For multi-GPU device_map=auto run `uv sync --extra infer-hf`."
        )
    try:
        model = AutoModelForCausalLM.from_pretrained(str(model_dir), **kwargs)
    except TypeError:
        kwargs.pop("dtype", None)
        kwargs["torch_dtype"] = dtype
        model = AutoModelForCausalLM.from_pretrained(str(model_dir), **kwargs)
    if cuda and "device_map" not in kwargs:
        model = model.to(device="cuda", dtype=dtype)
    if adapter_path is not None:
        from peft import PeftModel

        LOGGER.info("Attaching LoRA adapter %s", adapter_path)
        model = PeftModel.from_pretrained(model, str(adapter_path))
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    sanitize_generation_config(model)
    return tokenizer, model


def sanitize_generation_config(model) -> None:
    """Spark ships vLLM-style generation_config (top_k=-1, max_tokens) that breaks HF generate."""
    gc = getattr(model, "generation_config", None)
    if gc is None:
        return
    top_k = getattr(gc, "top_k", None)
    if top_k is not None:
        try:
            if int(top_k) < 0:
                gc.top_k = None
                LOGGER.info("Cleared generation_config.top_k=%s (not valid for transformers generate)", top_k)
        except (TypeError, ValueError):
            gc.top_k = None
    extra = getattr(gc, "max_tokens", None)
    if extra is not None and getattr(gc, "max_new_tokens", None) in {None, 0}:
        try:
            # Do not keep a million-token default; requests set max_new_tokens.
            delattr(gc, "max_tokens")
        except Exception:
            pass


def resolve_max_tokens(payload: dict[str, Any], default: int = 256) -> int:
    raw = payload.get("max_tokens")
    if raw is None:
        raw = payload.get("max_completion_tokens")
    try:
        value = int(raw) if raw is not None else default
    except (TypeError, ValueError):
        value = default
    return max(1, value)


def resolve_temperature(payload: dict[str, Any]) -> float:
    raw = payload.get("temperature")
    if raw is None:
        return 0.0
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 0.0


def build_generate_kwargs(*, max_new_tokens: int, temperature: float, tokenizer) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "max_new_tokens": max(1, int(max_new_tokens)),
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "do_sample": bool(temperature and temperature > 0),
    }
    if kwargs["do_sample"]:
        kwargs["temperature"] = max(float(temperature), 0.01)
    return kwargs


def render_chat_prompt(tokenizer, messages: list[dict[str, str]], *, enable_thinking: bool = False) -> str:
    if not (hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None)):
        parts = [f"{item.get('role', 'user')}: {item.get('content', '')}" for item in messages]
        return "\n".join(parts) + "\nassistant:"
    kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=enable_thinking, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def generate_chat(messages: list[dict[str, str]], *, max_tokens: int, temperature: float) -> str:
    result = generate_chat_result(messages, max_tokens=max_tokens, temperature=temperature)
    return result["text"]


def generate_chat_result(
    messages: list[dict[str, str]],
    *,
    max_tokens: int,
    temperature: float,
    enable_thinking: bool = False,
) -> dict[str, Any]:
    tokenizer = _STATE["tokenizer"]
    model = _STATE["model"]
    prompt = render_chat_prompt(tokenizer, messages, enable_thinking=enable_thinking)
    inputs = tokenizer(prompt, return_tensors="pt")
    device = _model_device(model)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    import torch
    gen_kwargs = build_generate_kwargs(max_new_tokens=max_tokens, temperature=temperature, tokenizer=tokenizer)
    LOGGER.info(
        "Generating max_new_tokens=%s do_sample=%s prompt_tokens=%s thinking=%s",
        gen_kwargs["max_new_tokens"],
        gen_kwargs["do_sample"],
        int(inputs["input_ids"].shape[-1]),
        enable_thinking,
    )
    with torch.no_grad():
        output = model.generate(
            **inputs,
            **gen_kwargs,
        )
    prompt_len = int(inputs["input_ids"].shape[-1])
    generated = output[0][prompt_len:]
    text = tokenizer.decode(generated, skip_special_tokens=True)
    completion_tokens = int(generated.shape[-1]) if hasattr(generated, "shape") else 0
    if completion_tokens == 0:
        LOGGER.warning(
            "Model generated 0 new tokens (prompt_tokens=%s eos_token_id=%s). "
            "Try enable_thinking=true or a higher max_tokens.",
            prompt_len,
            tokenizer.eos_token_id,
        )
    else:
        LOGGER.info("Generated completion_tokens=%s chars=%s", completion_tokens, len(text))
    return {
        "text": text,
        "prompt_tokens": prompt_len,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_len + completion_tokens,
    }


def _write_sse(handler: BaseHTTPRequestHandler, payload: dict[str, Any] | str) -> None:
    if isinstance(payload, str):
        body = f"data: {payload}\n\n"
    else:
        body = "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"
    handler.wfile.write(body.encode("utf-8"))
    handler.wfile.flush()


class OpenAIHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args: Any) -> None:
        LOGGER.info("%s - %s", self.address_string(), fmt % args)

    def do_GET(self) -> None:  # noqa: N802
        if self.path.split("?", 1)[0] != "/v1/models":
            _json(self, 404, {"error": {"message": "not found"}})
            return
        if not _authorize(self, _STATE.get("api_key")):
            _json(self, 401, {"error": {"message": "invalid api key"}})
            return
        _json(
            self,
            200,
            {
                "object": "list",
                "data": [{"id": _STATE["served_name"], "object": "model", "owned_by": "llm-tools"}],
            },
        )

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path not in {"/v1/chat/completions", "/v1/completions"}:
            _json(self, 404, {"error": {"message": "not found"}})
            return
        if not _authorize(self, _STATE.get("api_key")):
            _json(self, 401, {"error": {"message": "invalid api key"}})
            return
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            _json(self, 400, {"error": {"message": "invalid json"}})
            return
        messages = payload.get("messages")
        if not messages:
            prompt = payload.get("prompt") or ""
            messages = [{"role": "user", "content": str(prompt)}]
        template_kwargs = payload.get("chat_template_kwargs") if isinstance(payload.get("chat_template_kwargs"), dict) else {}
        enable_thinking = bool(payload.get("enable_thinking", template_kwargs.get("enable_thinking", False)))
        try:
            result = generate_chat_result(
                messages,
                max_tokens=resolve_max_tokens(payload),
                temperature=resolve_temperature(payload),
                enable_thinking=enable_thinking,
            )
        except Exception as exc:
            LOGGER.exception("Generation failed")
            _json(self, 500, {"error": {"message": str(exc)}})
            return
        model_name = payload.get("model") or _STATE["served_name"]
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())
        usage = {
            "prompt_tokens": result["prompt_tokens"],
            "completion_tokens": result["completion_tokens"],
            "total_tokens": result["total_tokens"],
        }
        if payload.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            _write_sse(
                self,
                {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_name,
                    "choices": [{"index": 0, "delta": {"role": "assistant", "content": result["text"]}, "finish_reason": None}],
                },
            )
            _write_sse(
                self,
                {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_name,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    "usage": usage,
                },
            )
            _write_sse(self, "[DONE]")
            return
        _json(
            self,
            200,
            {
                "id": completion_id,
                "object": "chat.completion",
                "created": created,
                "model": model_name,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": result["text"]},
                        "finish_reason": "stop",
                    }
                ],
                "usage": usage,
            },
        )


def serve_forever(host: str, port: int) -> None:
    httpd = ThreadingHTTPServer((host, int(port)), OpenAIHandler)
    LOGGER.info("transformers OpenAI server listening on %s:%s model=%s", host, port, _STATE.get("served_name"))
    httpd.serve_forever()


def configure_runtime(
    *,
    tokenizer,
    model,
    served_name: str,
    api_key: str | None,
) -> None:
    _STATE["tokenizer"] = tokenizer
    _STATE["model"] = model
    _STATE["served_name"] = served_name
    _STATE["api_key"] = api_key
