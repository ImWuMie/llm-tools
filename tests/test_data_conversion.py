from __future__ import annotations

from pathlib import Path

from common.data_format import DataOptions, parse_training_file, write_processed_jsonl


def test_default_txt_skips_comments_and_empty(tmp_path: Path) -> None:
    path = tmp_path / "sample.txt"
    path.write_text(
        "# comment\n\n你好\n你好！我是助手\n什么是LoRA？\nLoRA是低秩适配。\n",
        encoding="utf-8",
        newline="\n",
    )
    samples, report = parse_training_file(path, "default", DataOptions())
    assert report.samples == 2
    assert report.skipped_comments == 1
    assert report.skipped_empty >= 1
    assert samples[0][0]["role"] == "user"
    assert samples[0][1]["role"] == "assistant"


def test_system_prompt_inserted(tmp_path: Path) -> None:
    path = tmp_path / "sample.txt"
    path.write_text("user\nassistant\n", encoding="utf-8")
    samples, _report = parse_training_file(path, "default", DataOptions(system_prompt="你是助手"))
    assert samples[0][0] == {"role": "system", "content": "你是助手"}


def test_jsonl_sharegpt_alpaca(tmp_path: Path) -> None:
    jsonl = tmp_path / "a.jsonl"
    jsonl.write_text(
        '{"messages":[{"role":"user","content":"q"},{"role":"assistant","content":"a"}]}\n',
        encoding="utf-8",
    )
    samples, _ = parse_training_file(jsonl, "jsonl")
    assert samples[0][1]["content"] == "a"

    sharegpt = tmp_path / "s.jsonl"
    sharegpt.write_text(
        '{"conversations":[{"from":"human","value":"q"},{"from":"gpt","value":"a"}]}\n',
        encoding="utf-8",
    )
    samples, _ = parse_training_file(sharegpt, "sharegpt")
    assert [m["role"] for m in samples[0]] == ["user", "assistant"]

    alpaca = tmp_path / "p.jsonl"
    alpaca.write_text(
        '{"instruction":"翻译","input":"hi","output":"你好"}\n',
        encoding="utf-8",
    )
    samples, _ = parse_training_file(alpaca, "alpaca")
    assert "翻译" in samples[0][0]["content"]
    assert samples[0][1]["content"] == "你好"


def test_write_processed_jsonl(tmp_path: Path) -> None:
    out = tmp_path / "processed.jsonl"
    write_processed_jsonl([[{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]], out)
    assert out.exists()
    assert "messages" in out.read_text(encoding="utf-8")
