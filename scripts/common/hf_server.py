from __future__ import annotations

import json
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .logging_utils import setup_logging

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


def load_causal_lm(model_dir: Path, adapter_path: Path | None = None):
    from .rope_compat import apply_all_patches

    apply_all_patches()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    LOGGER.info("Loading transformers model from %s", model_dir)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    if adapter_path is not None:
        from peft import PeftModel

        LOGGER.info("Attaching LoRA adapter %s", adapter_path)
        model = PeftModel.from_pretrained(model, str(adapter_path))
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, model


def generate_chat(messages: list[dict[str, str]], *, max_tokens: int, temperature: float) -> str:
    tokenizer = _STATE["tokenizer"]
    model = _STATE["model"]
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        parts = [f"{item.get('role', 'user')}: {item.get('content', '')}" for item in messages]
        prompt = "\n".join(parts) + "\nassistant:"
    inputs = tokenizer(prompt, return_tensors="pt")
    if hasattr(model, "device"):
        inputs = {key: value.to(model.device) for key, value in inputs.items()}
    import torch

    do_sample = temperature is not None and temperature > 0
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max(1, int(max_tokens or 256)),
            do_sample=do_sample,
            temperature=max(temperature, 0.01) if do_sample else None,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated = output[0][inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(generated, skip_special_tokens=True)


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
        try:
            text = generate_chat(
                messages,
                max_tokens=int(payload.get("max_tokens") or 256),
                temperature=float(payload.get("temperature") or 0),
            )
        except Exception as exc:
            LOGGER.exception("Generation failed")
            _json(self, 500, {"error": {"message": str(exc)}})
            return
        model_name = payload.get("model") or _STATE["served_name"]
        _json(
            self,
            200,
            {
                "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model_name,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": text},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
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
