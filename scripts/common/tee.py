from __future__ import annotations

import argparse
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TextIO


REDACT_ENV = "LLM_TOOLS_REDACT"


def tee_bytes(src_fd: int, log_path: Path, *, console: bool, redact: str | None) -> None:
    """Copy a live byte stream to the log file and optionally the console."""
    return tee_bytes_to(
        src_fd,
        log_path,
        console_stream=sys.stdout if console else None,
        redact=redact,
    )


def tee_bytes_to(
    src_fd: int,
    log_path: Path,
    *,
    console_stream: TextIO | None,
    redact: str | None,
) -> None:
    """Copy a live byte stream to the log file and optionally the console."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", buffering=1) as log_fp:
        while True:
            try:
                chunk = os.read(src_fd, 4096)
            except OSError:
                break
            if not chunk:
                break
            text = chunk.decode("utf-8", errors="replace")
            if redact:
                text = text.replace(redact, "***")
            log_fp.write(text)
            log_fp.flush()
            if console_stream is not None:
                console_stream.write(text)
                console_stream.flush()


def run_command_teed(
    cmd: Sequence[str],
    log_path: Path,
    *,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
    console: bool = True,
    redact: str | None = None,
    console_stream: TextIO | None = None,
) -> int:
    proc = subprocess.Popen(
        list(cmd),
        cwd=str(cwd) if cwd is not None else None,
        env=dict(env) if env is not None else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=0,
    )
    assert proc.stdout is not None
    try:
        stream = console_stream if console_stream is not None else (sys.stdout if console else None)
        tee_bytes_to(proc.stdout.fileno(), log_path, console_stream=stream, redact=redact)
    finally:
        proc.stdout.close()
    return int(proc.wait())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Tee a command to a log file and the console.")
    parser.add_argument("--log", required=True)
    parser.add_argument("--no-console", action="store_true")
    parser.add_argument("cmd", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    cmd = [item for item in args.cmd if item != "--"]
    if not cmd:
        parser.error("missing command after --")
    console = not args.no_console
    redact = os.environ.get(REDACT_ENV) or None
    child_env = os.environ.copy()
    child_env.pop(REDACT_ENV, None)
    return run_command_teed(
        cmd,
        Path(args.log),
        console=console,
        redact=redact,
        env=child_env,
    )


if __name__ == "__main__":
    sys.exit(main())
