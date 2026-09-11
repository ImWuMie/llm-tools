#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from common.bootstrap import ensure_sys_path

ensure_sys_path()

from common.env import ConfigError, apply_runtime_env, load_app_config
from common.logging_utils import setup_logging
from common.vllm_launcher import VLLMLaunchError, cli_windows_backend, start_vllm_server

LOGGER = setup_logging("llm_tools.start_vllm")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start an OpenAI-compatible vLLM server for the base model.")
    parser.add_argument("--foreground", action="store_true", help="Run in the foreground (default).")
    parser.add_argument("--daemon", action="store_true", help="Run in the background and write a PID file.")
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", default=None)
    parser.add_argument("--wsl", action="store_true", help="On Windows, delegate to WSL2.")
    parser.add_argument("--docker", action="store_true", help="On Windows, start docker compose service vllm.")
    parser.add_argument(
        "--native",
        action="store_true",
        help="On Windows, use an already-installed community vLLM wheel in this Python env.",
    )
    parser.add_argument("--force-native", action="store_true", help="Skip Windows WSL/Docker redirection.")
    parser.add_argument("extra", nargs=argparse.REMAINDER, help="Extra args forwarded to vLLM after `--`.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config = load_app_config()
        apply_runtime_env(config)
        config.require_keys(["MODEL_DIR", "VLLM_HOST", "VLLM_PORT"])
        config.override(VLLM_HOST=args.host, VLLM_PORT=str(args.port) if args.port else None)
        model_dir = Path(args.model_dir) if args.model_dir else config.require_path("MODEL_DIR")
        if not model_dir.exists():
            raise ConfigError(
                f"MODEL_DIR does not exist: {model_dir}. "
                "Run `uv run python scripts/download_model.py --source auto --update-env` first, "
                "or point MODEL_DIR at a local snapshot."
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
            model_path=model_dir,
            service_name="vllm",
            daemon=daemon,
            extra_args=extra,
            windows_backend=backend,
            force_native=force_native,
        )
    except (ConfigError, VLLMLaunchError) as exc:
        LOGGER.error("%s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
