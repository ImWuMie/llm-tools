#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys

from common.bootstrap import ensure_sys_path

ensure_sys_path()

from common.env import ConfigError, apply_runtime_env, load_app_config
from common.logging_utils import setup_logging
from common.vllm_launcher import VLLMLaunchError, cli_windows_backend, detect_trained_mode, start_vllm_server

LOGGER = setup_logging("llm_tools.start_vllm_trained")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start vLLM for a trained / LoRA / merged checkpoint.")
    parser.add_argument("--foreground", action="store_true")
    parser.add_argument("--daemon", action="store_true")
    parser.add_argument("--mode", choices=["auto", "lora", "merged"], default=None)
    parser.add_argument("--wsl", action="store_true")
    parser.add_argument("--docker", action="store_true")
    parser.add_argument(
        "--native",
        action="store_true",
        help="On Windows, use an already-installed community vLLM wheel in this Python env.",
    )
    parser.add_argument("--force-native", action="store_true")
    parser.add_argument("--engine", choices=["auto", "vllm", "hf"], default=None)
    parser.add_argument("extra", nargs=argparse.REMAINDER)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config = load_app_config()
        apply_runtime_env(config)
        config.require_keys(["VLLM_HOST", "VLLM_PORT", "CHECKPOINT_PATH", "BASE_MODEL", "ADAPTER_PATH"])
        mode, model_path, adapter_path = detect_trained_mode(config, requested=args.mode)
        LOGGER.info("Detected trained model mode=%s model=%s adapter=%s", mode, model_path, adapter_path)
        lora_modules = None
        if mode == "lora":
            name = config.get("TRAINED_LORA_NAME") or "trained"
            lora_modules = {name: adapter_path}
            LOGGER.info(
                "Serving base model with LoRA adapter `%s`. "
                "OpenAI requests should use model=%s (or the served base name).",
                adapter_path,
                name,
            )
        extra = [item for item in args.extra if item != "--"]
        backend, force_native = cli_windows_backend(
            wsl=args.wsl,
            docker=args.docker,
            native=args.native,
            force_native=args.force_native,
        )
        daemon = bool(args.daemon and not args.foreground)
        return start_vllm_server(
            config=config,
            model_path=model_path,
            service_name="vllm_trained",
            daemon=daemon,
            extra_args=extra,
            lora_modules=lora_modules,
            windows_backend=backend,
            force_native=force_native,
            engine=args.engine,
        )
    except (ConfigError, VLLMLaunchError) as exc:
        LOGGER.error("%s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
