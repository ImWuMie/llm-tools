from __future__ import annotations

from pathlib import Path

from common.data_format import DataOptions
from common.hub_data import looks_like_hub_id, records_to_samples
from common.license_check import inspect_license
from common.metrics import assistant_turns, exact_match, token_f1


def test_metrics_basic() -> None:
    assert exact_match("a", "a") == 1.0
    assert exact_match("a", "b") == 0.0
    assert token_f1("hello world", "hello there") > 0


def test_assistant_turns_multi() -> None:
    messages = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "a2"},
    ]
    pairs = assistant_turns(messages)
    assert len(pairs) == 2
    assert pairs[0][1] == "a1"
    assert pairs[1][0][-1]["content"] == "q2"


def test_looks_like_hub_id(tmp_path: Path) -> None:
    assert looks_like_hub_id("org/name") is True
    local = tmp_path / "sample.txt"
    local.write_text("x", encoding="utf-8")
    assert looks_like_hub_id(str(local)) is False


def test_records_to_samples_alpaca() -> None:
    samples = records_to_samples(
        [{"instruction": "hi", "input": "", "output": "hello"}],
        DataOptions(),
    )
    assert samples[0][0]["role"] == "user"
    assert samples[0][1]["role"] == "assistant"


def test_inspect_license_from_readme(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("---\nlicense: mit\n---\n# model\n", encoding="utf-8")
    report = inspect_license(tmp_path)
    assert report.license == "mit"
    assert report.commercial_ok is True
