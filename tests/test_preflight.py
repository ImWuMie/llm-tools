from __future__ import annotations

import json
from pathlib import Path

from common.preflight import run_preflight, vllm_likely_supported


def test_qwen_is_vllm_supported() -> None:
    assert vllm_likely_supported({"model_type": "qwen2"}) is True


def test_spark_custom_code_is_hf() -> None:
    config = {
        "model_type": "spark2_5",
        "architectures": ["Spark2_5ForCausalLM"],
        "auto_map": {"AutoModelForCausalLM": "modeling_spark.Spark2_5ForCausalLM"},
    }
    assert vllm_likely_supported(config, plugin_available=False) is False


def test_spark_with_plugin_is_vllm() -> None:
    config = {
        "model_type": "spark2_5",
        "architectures": ["Spark2_5ForCausalLM"],
        "auto_map": {"AutoModelForCausalLM": "modeling_spark.Spark2_5ForCausalLM"},
    }
    assert vllm_likely_supported(config, plugin_available=True) is True


def test_run_preflight_on_tmp_model(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "spark2_5",
                "architectures": ["Spark2_5ForCausalLM"],
                "auto_map": {"AutoConfig": "configuration_spark.Spark2_5Config"},
                "hidden_size": 2560,
                "num_hidden_layers": 36,
                "vocab_size": 1000,
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "tokenizer.json").write_text("{}", encoding="utf-8")
    (tmp_path / "model.safetensors").write_bytes(b"not-empty")
    report = run_preflight(tmp_path, max_model_len=2048)
    assert report.engine_hint == "hf"
    assert report.ok is True


def test_run_preflight_spark_plugin_switches_engine(tmp_path: Path, monkeypatch) -> None:
    from common import preflight as preflight_mod

    monkeypatch.setattr(preflight_mod, "spark_plugin_installed", lambda: True)
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "spark2_5",
                "architectures": ["Spark2_5ForCausalLM"],
                "auto_map": {"AutoConfig": "configuration_spark.Spark2_5Config"},
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "tokenizer.json").write_text("{}", encoding="utf-8")
    (tmp_path / "model.safetensors").write_bytes(b"not-empty")
    report = run_preflight(tmp_path, max_model_len=1024)
    assert report.engine_hint == "vllm"
