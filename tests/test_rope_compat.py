from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from common.env import AppConfig
from common.rope_compat import (
    apply_rope_to_config,
    is_per_layer_rope,
    make_validate_rope_wrapper,
    normalize_rope_parameters,
    replace_class_validators,
)
from common.vllm_launcher import build_vllm_command
from common.windows_vllm_runtime import TOOLCHAIN_ONLY_ENV, serving_child_env

NESTED = {
    "full_attention": {"partial_rotary_factor": 0.25, "rope_theta": 5000000},
    "sliding_attention": {"partial_rotary_factor": 1.0, "rope_theta": 10000},
}


def test_nested_rope_stays_nested_and_drops_float_siblings() -> None:
    assert is_per_layer_rope(NESTED) is True
    out = normalize_rope_parameters(NESTED)
    assert "rope_theta" not in out or isinstance(out.get("rope_theta"), dict)
    assert out["full_attention"]["rope_type"] == "default"
    assert out["sliding_attention"]["rope_theta"] == 10000
    injected = dict(NESTED)
    injected["rope_theta"] = 10000.0
    injected["rope_type"] = "default"
    cleaned = normalize_rope_parameters(injected)
    assert all(isinstance(value, dict) for value in cleaned.values())
    assert "rope_theta" not in cleaned
    assert normalize_rope_parameters(10000.0) == {"rope_type": "default", "rope_theta": 10000.0}


def test_apply_rope_to_config_makes_validate_rope_safe() -> None:
    config = SimpleNamespace(
        rope_parameters={**NESTED, "rope_theta": 10000.0},
        layer_types=["full_attention", "sliding_attention", "full_attention"],
        ignore_keys_at_rope_validation=set(),
    )

    def original_validate_rope(self):  # noqa: ANN001
        rope_parameters_dict = self.rope_parameters
        if getattr(self, "layer_types", None) is not None and not set(rope_parameters_dict.keys()).isdisjoint(
            self.layer_types
        ):
            pass
        else:
            rope_parameters_dict = {"full_attention": rope_parameters_dict}
        for rope_parameters in rope_parameters_dict.values():
            if rope_parameters is None:
                continue
            rope_type = rope_parameters.get("rope_type", rope_parameters.get("type", "default"))
            rope_parameters["rope_type"] = rope_type

    apply_rope_to_config(config)
    original_validate_rope(config)
    assert config.rope_parameters["full_attention"]["rope_type"] == "default"


def test_class_validator_list_is_replaced() -> None:
    seen: list[str] = []

    def original(self):  # noqa: ANN001
        seen.append("original")
        self.rope_parameters.get("rope_type")

    class Holder:
        __class_validators__ = [original]
        validate_rope = original

    wrapped = make_validate_rope_wrapper(original)
    assert replace_class_validators(Holder, original, wrapped) == 1
    cfg = SimpleNamespace(rope_parameters=10000.0, ignore_keys_at_rope_validation=set())
    Holder.__class_validators__[0](cfg)
    assert "original" in seen
    assert isinstance(cfg.rope_parameters, dict)
    assert cfg.rope_parameters["rope_type"] == "default"


def test_vllm_command_uses_patched_launcher(tmp_path: Path) -> None:
    config = AppConfig(
        values={
            "VLLM_HOST": "0.0.0.0",
            "VLLM_PORT": "8000",
            "TENSOR_PARALLEL_SIZE": "1",
            "GPU_MEMORY_UTILIZATION": "0.9",
            "MAX_MODEL_LEN": "4096",
            "DTYPE": "auto",
            "VLLM_MAX_NUM_SEQS": "16",
            "VLLM_TRUST_REMOTE_CODE": "1",
        },
        env_path=tmp_path / ".env",
        project_root=tmp_path,
    )
    cmd = build_vllm_command("python", tmp_path / "model", config)
    assert any(str(part).endswith("run_vllm_server.py") for part in cmd)
    assert "vllm.entrypoints.openai.api_server" not in cmd


def test_serving_child_env_strips_vllm_host(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_HOST", "0.0.0.0")
    monkeypatch.setenv("VLLM_PORT", "8000")
    monkeypatch.setenv("VLLM_SPARK_PLUGIN", "1")
    env = serving_child_env()
    for key in ("VLLM_HOST", "VLLM_PORT", "VLLM_SPARK_PLUGIN"):
        assert key in TOOLCHAIN_ONLY_ENV
        assert key not in env
