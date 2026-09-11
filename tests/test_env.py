from __future__ import annotations

from pathlib import Path

from common.env import ensure_env_file, load_app_config, upsert_env_key
from common.secrets import mask_secret, redact_command, redact_mapping


def test_ensure_env_file_copies_example(tmp_path: Path) -> None:
    example = tmp_path / ".env_example"
    example.write_text("MODEL_ID=demo\nHF_TOKEN=secret-token\n", encoding="utf-8")
    path = ensure_env_file(tmp_path)
    assert path.exists()
    assert "MODEL_ID=demo" in path.read_text(encoding="utf-8")


def test_load_app_config_and_redact(tmp_path: Path) -> None:
    (tmp_path / ".env_example").write_text("HF_TOKEN=abcdefghijk\nMODEL_ID=x\n", encoding="utf-8")
    config = load_app_config(tmp_path)
    redacted = config.redacted()
    assert "abcdefghijk" not in str(redacted)
    assert redacted["HF_TOKEN"].startswith("abcd")


def test_upsert_env_key_preserves_other_lines(tmp_path: Path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text("A=1\nHF_TOKEN=super-secret\nB=2\n", encoding="utf-8")
    upsert_env_key(env_path, "B", "3")
    text = env_path.read_text(encoding="utf-8")
    assert "A=1" in text
    assert "B=3" in text
    assert "HF_TOKEN=super-secret" in text


def test_mask_and_redact_command() -> None:
    assert mask_secret("short") == "***"
    cmd = ["python", "-m", "vllm", "--api-key", "sk-secret", "--token=abc"]
    redacted = redact_command(cmd)
    assert "sk-secret" not in redacted
    assert redact_mapping({"HF_TOKEN": "abcdefghijk", "MODEL_ID": "x"})["MODEL_ID"] == "x"
