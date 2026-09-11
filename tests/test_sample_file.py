from __future__ import annotations

from pathlib import Path

from common.bootstrap import PROJECT_ROOT
from common.data_format import DataOptions, parse_training_file


def test_repo_sample_txt_converts() -> None:
    path = PROJECT_ROOT / "training" / "data" / "sample.txt"
    samples, report = parse_training_file(path, "default", DataOptions())
    assert report.utf8_ok
    assert report.samples == 3
    assert samples[0][0]["role"] == "user"
    assert samples[0][1]["role"] == "assistant"
    assert all(turn["content"] for sample in samples for turn in sample)
