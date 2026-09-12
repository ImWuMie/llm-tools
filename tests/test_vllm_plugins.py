from __future__ import annotations

from common.vllm_plugins import is_spark_architecture, plugin_covers_architecture, spark_plugin_installed


def test_spark_architecture_detection() -> None:
    assert is_spark_architecture({"model_type": "spark2_5", "architectures": ["Spark2_5ForCausalLM"]})
    assert not is_spark_architecture({"model_type": "qwen2"})


def test_plugin_cover_honors_flag() -> None:
    config = {"model_type": "spark2_5", "architectures": ["Spark2_5ForCausalLM"]}
    assert plugin_covers_architecture(config, plugin_available=True) is True
    assert plugin_covers_architecture(config, plugin_available=False) is False
    assert plugin_covers_architecture({"model_type": "qwen2"}, plugin_available=True) is False


def test_env_force_spark_plugin(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_SPARK_PLUGIN", "1")
    assert spark_plugin_installed() is True
    monkeypatch.setenv("VLLM_SPARK_PLUGIN", "0")
    assert spark_plugin_installed() is False
