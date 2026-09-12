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

LOGGER = setup_logging("llm_tools.chat")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal OpenAI-compatible chat client.")
    parser.add_argument("--prompt", default="Say hello in one sentence.")
    parser.add_argument("--model", default=None)
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--system", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_app_config()
    host = config.get("VLLM_HOST") or "127.0.0.1"
    if host in {"0.0.0.0", "::"}:
        host = "127.0.0.1"
    port = config.get("VLLM_PORT") or "8000"
    api_key = config.get("VLLM_API_KEY") or os.environ.get("OPENAI_API_KEY") or "sk-local"
    model = args.model or config.get("VLLM_SERVED_MODEL_NAME") or config.get("TRAINED_LORA_NAME") or "default"
    base_url = f"http://{host}:{port}/v1"
    LOGGER.info("Connecting to %s model=%s api_key=%s", base_url, model, mask_secret(api_key))

    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key=api_key)
    messages = []
    if args.system:
        messages.append({"role": "system", "content": args.system})
    messages.append({"role": "user", "content": args.prompt})

    if args.stream:
        stream = client.chat.completions.create(model=model, messages=messages, stream=True)
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            print(delta, end="", flush=True)
        print()
        return 0

    completion = client.chat.completions.create(model=model, messages=messages)
    print(completion.choices[0].message.content)
    return 0


if __name__ == "__main__":
    sys.exit(main())
