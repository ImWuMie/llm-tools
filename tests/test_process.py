from __future__ import annotations

import sys
from pathlib import Path

from common.process import sanitize_omp_env, start_process


def test_sanitize_omp_env_drops_invalid_values() -> None:
    env = {"OMP_NUM_THREADS": "", "KEEP": "1"}
    sanitize_omp_env(env)
    assert "OMP_NUM_THREADS" not in env
    assert env["KEEP"] == "1"
    env = {"OMP_NUM_THREADS": "0"}
    sanitize_omp_env(env)
    assert "OMP_NUM_THREADS" not in env
    env = {"OMP_NUM_THREADS": "auto"}
    sanitize_omp_env(env)
    assert "OMP_NUM_THREADS" not in env
    env = {"OMP_NUM_THREADS": "8"}
    sanitize_omp_env(env)
    assert env["OMP_NUM_THREADS"] == "8"


def test_start_process_resets_log_file(tmp_path: Path) -> None:
    log_path = tmp_path / "vllm.log"
    log_path.write_text("old crash\nAttributeError\n", encoding="utf-8")
    proc = start_process(
        [sys.executable, "-c", "print('fresh-start')"],
        cwd=tmp_path,
        log_path=log_path,
        daemon=True,
    )
    proc.wait(timeout=10)
    text = log_path.read_text(encoding="utf-8")
    assert "old crash" not in text
    assert "fresh-start" in text
