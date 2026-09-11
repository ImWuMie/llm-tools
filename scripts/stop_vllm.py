#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from common.bootstrap import ensure_sys_path

ensure_sys_path()

from common.env import ConfigError, load_app_config
from common.logging_utils import setup_logging
from common.process import is_pid_running, pid_file, read_pid, stop_pid

LOGGER = setup_logging("llm_tools.stop_vllm")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stop vLLM processes started by this toolchain.")
    parser.add_argument("--service", choices=["vllm", "vllm_trained", "all"], default="all")
    parser.add_argument("--pid-file", default=None)
    parser.add_argument("--timeout", type=float, default=20.0)
    return parser.parse_args()


def stop_one(path: Path, timeout: float) -> int:
    pid = read_pid(path)
    if pid is None:
        LOGGER.info("No PID file at %s", path)
        return 0
    if not is_pid_running(pid):
        LOGGER.info("Stale PID file %s (pid=%s). Removing.", path, pid)
        path.unlink(missing_ok=True)
        return 0
    LOGGER.info("Stopping pid=%s from %s", pid, path)
    stop_pid(pid, timeout=timeout)
    path.unlink(missing_ok=True)
    return 0


def main() -> int:
    args = parse_args()
    try:
        config = load_app_config()
        pid_dir = config.require_path("PID_DIR")
        if args.pid_file:
            return stop_one(Path(args.pid_file), args.timeout)
        names = ["vllm", "vllm_trained"] if args.service == "all" else [args.service]
        code = 0
        for name in names:
            code = max(code, stop_one(pid_file(pid_dir, name), args.timeout))
        return code
    except ConfigError as exc:
        LOGGER.error("%s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
