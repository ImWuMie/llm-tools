from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

from common.process import wait_for_or_exit
from common.windows_vllm_runtime import (
    TOOLCHAIN_ONLY_ENV,
    default_native_health_timeout,
    install_xgrammar_stub,
    log_tail,
    native_windows_child_env,
    serving_child_env,
)


def test_native_child_env_strips_toolchain_keys_and_defaults_sampler(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("VLLM_WINDOWS_BACKEND", "native")
    monkeypatch.setenv("VLLM_HEALTH_TIMEOUT", "180")
    monkeypatch.delenv("VLLM_USE_FLASHINFER_SAMPLER", raising=False)
    monkeypatch.setenv("VLLM_HOST", "0.0.0.0")
    env = native_windows_child_env()
    for key in TOOLCHAIN_ONLY_ENV:
        assert key not in env
    assert env["VLLM_USE_FLASHINFER_SAMPLER"] == "0"
    assert "sitecustomize" not in Path(env["PYTHONPATH"].split(os.pathsep)[0]).name
    assert (Path(env["PYTHONPATH"].split(os.pathsep)[0]) / "sitecustomize.py").is_file()


def test_serving_child_env_strips_vllm_host(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_HOST", "0.0.0.0")
    monkeypatch.setenv("VLLM_PORT", "8000")
    monkeypatch.setenv("VLLM_SPARK_PLUGIN", "1")
    env = serving_child_env()
    for key in ("VLLM_HOST", "VLLM_PORT", "VLLM_SPARK_PLUGIN"):
        assert key in TOOLCHAIN_ONLY_ENV
        assert key not in env


def test_xgrammar_stub_exposes_grammar_matcher() -> None:
    install_xgrammar_stub("unit-test")
    import xgrammar

    assert hasattr(xgrammar, "GrammarMatcher")
    assert getattr(xgrammar, "_llm_tools_stub", False)


def test_health_timeout_default() -> None:
    assert default_native_health_timeout(None) == 600.0
    assert default_native_health_timeout(180) == 180.0


def test_wait_for_or_exit_detects_crash() -> None:
    proc = SimpleNamespace(returncode=1)

    def poll() -> int:
        return 1

    proc.poll = poll  # type: ignore[attr-defined]
    assert wait_for_or_exit(proc, lambda: False, timeout=0.2, interval=0.05, description="unit") == "exited"


def test_log_tail_redacts_secret(tmp_path: Path) -> None:
    path = tmp_path / "vllm.log"
    path.write_text("api-key sk-secret-value appeared\n", encoding="utf-8")
    text = log_tail(path, secret="sk-secret-value")
    assert "sk-secret-value" not in text
    assert "***" in text
