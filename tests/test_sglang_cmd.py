from __future__ import annotations

from pathlib import Path

from common.env import AppConfig
from start_sglang import build_sglang_command


def test_sglang_command_uses_vllm_env(tmp_path: Path) -> None:
    config = AppConfig(
        values={
            "VLLM_HOST": "0.0.0.0",
            "VLLM_PORT": "8000",
            "TENSOR_PARALLEL_SIZE": "1",
            "GPU_MEMORY_UTILIZATION": "0.9",
            "MAX_MODEL_LEN": "8192",
            "DTYPE": "auto",
            "VLLM_TRUST_REMOTE_CODE": "1",
            "VLLM_API_KEY": "sk-local",
            "SGLANG_TOOL_CALL_PARSER": "qwen25",
            "SGLANG_REASONING_PARSER": "qwen3",
        },
        env_path=tmp_path / ".env",
        project_root=tmp_path,
    )
    cmd = build_sglang_command("python", tmp_path / "model", config, ["--chunked-prefill-size", "8192"])
    assert "sglang.launch_server" in cmd
    assert "--model-path" in cmd
    assert "--context-length" in cmd
    assert "8192" in cmd
    assert "--mem-fraction-static" in cmd
    assert "--tool-call-parser" in cmd
    assert "qwen25" in cmd
    assert "--reasoning-parser" in cmd
    assert "--api-key" in cmd
    assert cmd[-2:] == ["--chunked-prefill-size", "8192"]
