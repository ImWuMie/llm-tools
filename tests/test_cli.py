from __future__ import annotations

import runpy
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


@pytest.mark.parametrize(
    "script",
    [
        "download_model.py",
        "start_vllm.py",
        "start_vllm_trained.py",
        "train.py",
        "stop_vllm.py",
        "merge_lora.py",
        "eval.py",
        "selfcheck.py",
        "install_vllm_windows.py",
    ],
)
def test_cli_help(script: str, capsys: pytest.CaptureFixture[str]) -> None:
    sys.argv = [script, "--help"]
    with pytest.raises(SystemExit) as exc:
        runpy.run_path(str(SCRIPTS / script), run_name="__main__")
    assert exc.value.code == 0
    captured = capsys.readouterr()
    assert "usage" in captured.out.lower() or "usage" in captured.err.lower()
