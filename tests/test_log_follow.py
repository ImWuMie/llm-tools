from __future__ import annotations

from io import StringIO
from pathlib import Path
from types import SimpleNamespace

from common.log_follow import LogFollower
from common.process import wait_for_or_exit


def test_log_follower_streams_new_lines_and_redacts(tmp_path: Path) -> None:
    path = tmp_path / "vllm.log"
    path.write_text("old line\n", encoding="utf-8")
    buf = StringIO()
    follower = LogFollower(path, secret="sk-secret", stream=buf)
    path.write_text("old line\napi-key sk-secret boom\npartial", encoding="utf-8")
    follower.poll()
    text = buf.getvalue()
    assert "old line" not in text
    assert "sk-secret" not in text
    assert "***" in text
    assert "partial" not in text
    follower.close()
    assert "partial" in buf.getvalue()


def test_wait_for_or_exit_follows_log(tmp_path: Path) -> None:
    path = tmp_path / "vllm.log"
    path.write_text("", encoding="utf-8")
    proc = SimpleNamespace(returncode=1)
    proc.poll = lambda: 1  # type: ignore[attr-defined]
    path.write_text("engine died\n", encoding="utf-8")
    assert wait_for_or_exit(proc, lambda: False, timeout=0.4, interval=0.05, log_path=path) == "exited"
