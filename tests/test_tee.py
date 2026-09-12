from __future__ import annotations

import sys
from io import StringIO
from pathlib import Path

from common.tee import run_command_teed


def test_run_command_teed_writes_log_and_console(tmp_path: Path) -> None:
    log_path = tmp_path / "vllm.log"
    buf = StringIO()
    code = run_command_teed(
        [sys.executable, "-c", "import sys; print('hello-tee', flush=True); print('err-tee', file=sys.stderr, flush=True)"],
        log_path,
        cwd=tmp_path,
        console_stream=buf,
    )
    assert code == 0
    text = log_path.read_text(encoding="utf-8")
    assert "hello-tee" in text
    assert "err-tee" in text
    assert "hello-tee" in buf.getvalue()
    assert "err-tee" in buf.getvalue()


def test_tee_redacts_secret(tmp_path: Path) -> None:
    log_path = tmp_path / "vllm.log"
    buf = StringIO()
    code = run_command_teed(
        [sys.executable, "-c", "print('token sk-secret-value')"],
        log_path,
        cwd=tmp_path,
        console_stream=buf,
        redact="sk-secret-value",
    )
    assert code == 0
    text = log_path.read_text(encoding="utf-8")
    assert "sk-secret-value" not in text
    assert "***" in text
    assert "sk-secret-value" not in buf.getvalue()
