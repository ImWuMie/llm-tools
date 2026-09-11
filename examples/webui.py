#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from common.env import load_app_config
from common.logging_utils import setup_logging
from common.secrets import mask_secret

LOGGER = setup_logging("llm_tools.webui")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gradio chat UI for the local OpenAI-compatible server.")
    parser.add_argument("--model", default=None)
    parser.add_argument("--share", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        import gradio as gr
    except ImportError:
        LOGGER.error("Gradio is missing. Run `uv sync --extra webui`.")
        return 1

    config = load_app_config()
    host = config.get("VLLM_HOST") or "127.0.0.1"
    if host in {"0.0.0.0", "::"}:
        host = "127.0.0.1"
    port = config.get("VLLM_PORT") or "8000"
    api_key = config.get("VLLM_API_KEY") or os.environ.get("OPENAI_API_KEY") or "sk-local"
    default_model = args.model or config.get("VLLM_SERVED_MODEL_NAME") or config.get("TRAINED_LORA_NAME") or "default"
    base_url = f"http://{host}:{port}/v1"
    LOGGER.info("WebUI -> %s model=%s api_key=%s", base_url, default_model, mask_secret(api_key))

    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key=api_key)

    def chat(message: str, history: list[dict[str, str]], model_name: str):
        messages = []
        for item in history:
            messages.append({"role": item["role"], "content": item["content"]})
        messages.append({"role": "user", "content": message})
        completion = client.chat.completions.create(model=model_name or default_model, messages=messages, temperature=0.7)
        return completion.choices[0].message.content or ""

    demo = gr.ChatInterface(
        fn=chat,
        additional_inputs=[gr.Textbox(value=default_model, label="model")],
        title="LLM Tools",
        description=f"Talking to {base_url}. Start the server first.",
    )
    demo.launch(share=bool(args.share))
    return 0


if __name__ == "__main__":
    sys.exit(main())
