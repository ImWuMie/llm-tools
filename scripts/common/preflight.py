from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .logging_utils import setup_logging
from .validate_model import validate_local_model
from .vllm_plugins import is_spark_architecture, plugin_covers_architecture, spark_plugin_installed

LOGGER = setup_logging("llm_tools.preflight")

KNOWN_VLLM_TYPES = {
    "llama",
    "mistral",
    "mixtral",
    "qwen2",
    "qwen2_moe",
    "qwen3",
    "qwen3_moe",
    "gemma",
    "gemma2",
    "gemma3",
    "phi",
    "phi3",
    "phimoe",
    "gpt2",
    "gpt_neox",
    "gptj",
    "falcon",
    "bloom",
    "opt",
    "baichuan",
    "chatglm",
    "internlm",
    "internlm2",
    "deepseek",
    "deepseek_v2",
    "deepseek_v3",
    "glm4",
    "cohere",
    "command-r",
    "starcoder2",
    "stablelm",
}


@dataclass
class PreflightReport:
    model_dir: str
    ok: bool = True
    engine_hint: str = "vllm"
    model_type: str | None = None
    architecture: str | None = None
    custom_code: bool = False
    estimated_weight_gb: float | None = None
    estimated_kv_gb: float | None = None
    estimated_total_gb: float | None = None
    gpu_memory_gb: float | None = None
    problems: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_dir": self.model_dir,
            "ok": self.ok,
            "engine_hint": self.engine_hint,
            "model_type": self.model_type,
            "architecture": self.architecture,
            "custom_code": self.custom_code,
            "estimated_weight_gb": self.estimated_weight_gb,
            "estimated_kv_gb": self.estimated_kv_gb,
            "estimated_total_gb": self.estimated_total_gb,
            "gpu_memory_gb": self.gpu_memory_gb,
            "problems": self.problems,
            "warnings": self.warnings,
        }


def read_model_config(model_dir: Path) -> dict[str, Any]:
    path = model_dir / "config.json"
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def read_index_metadata(model_dir: Path) -> dict[str, Any]:
    path = model_dir / "model.safetensors.index.json"
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    metadata = payload.get("metadata") if isinstance(payload, dict) else None
    return metadata if isinstance(metadata, dict) else {}


def _dtype_bytes(dtype: str | None) -> int:
    text = (dtype or "bf16").lower()
    if "8" in text or "int4" in text or "4bit" in text:
        return 1
    if "32" in text or "float32" in text or "fp32" in text:
        return 4
    return 2


def estimate_parameters(config: dict[str, Any], metadata: dict[str, Any]) -> int | None:
    total = metadata.get("total_parameters")
    if isinstance(total, int) and total > 0:
        return total
    hidden = int(config.get("hidden_size") or 0)
    layers = int(config.get("num_hidden_layers") or 0)
    vocab = int(config.get("vocab_size") or 0)
    intermediate = int(config.get("intermediate_size") or hidden * 4)
    if hidden <= 0 or layers <= 0:
        return None
    attn = 4 * hidden * hidden
    mlp = 3 * hidden * intermediate
    embed = vocab * hidden
    return layers * (attn + mlp) + embed


def estimate_memory_gb(
    config: dict[str, Any],
    metadata: dict[str, Any],
    *,
    max_model_len: int,
    dtype: str | None,
) -> tuple[float | None, float | None, float | None]:
    params = estimate_parameters(config, metadata)
    if not params:
        return None, None, None
    width = _dtype_bytes(dtype or str(config.get("dtype") or "bf16"))
    weights = params * width
    layers = int(config.get("num_hidden_layers") or 0)
    kv_heads = int(config.get("num_key_value_heads") or config.get("num_attention_heads") or 0)
    head_dim = int(config.get("head_dim") or 0)
    if head_dim <= 0:
        heads = int(config.get("num_attention_heads") or 0)
        hidden = int(config.get("hidden_size") or 0)
        if heads:
            head_dim = hidden // heads
    kv = layers * 2 * kv_heads * max(head_dim, 1) * max(max_model_len, 1) * width
    total = (weights + kv) * 1.25
    return weights / 1024**3, kv / 1024**3, total / 1024**3


def gpu_memory_gb() -> float | None:
    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return float(props.total_memory) / 1024**3
    except Exception:
        pass
    return None


def nvidia_smi_memory_gb() -> float | None:
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            text=True,
            timeout=5,
            stderr=subprocess.DEVNULL,
        )
        first = output.strip().splitlines()[0].strip()
        value = float(first.split()[0])
        return value / 1024.0 if value > 64 else value
    except Exception:
        return None


def torch_cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def vllm_likely_supported(config: dict[str, Any], *, plugin_available: bool | None = None) -> bool:
    model_type = str(config.get("model_type") or "").strip().lower()
    if model_type in KNOWN_VLLM_TYPES:
        return True
    if plugin_covers_architecture(config, plugin_available=plugin_available):
        return True
    auto_map = config.get("auto_map")
    if isinstance(auto_map, dict) and auto_map:
        return False
    architectures = config.get("architectures") or []
    if architectures and model_type not in KNOWN_VLLM_TYPES:
        return False
    return bool(model_type)


def run_preflight(
    model_dir: Path,
    *,
    max_model_len: int = 4096,
    dtype: str | None = None,
    gpu_memory_utilization: float = 0.9,
) -> PreflightReport:
    report = PreflightReport(model_dir=str(model_dir))
    problems = validate_local_model(model_dir)
    report.problems.extend(problems)
    config = read_model_config(model_dir)
    metadata = read_index_metadata(model_dir)
    report.model_type = str(config.get("model_type") or "") or None
    architectures = config.get("architectures") or []
    if architectures:
        report.architecture = str(architectures[0])
    report.custom_code = bool(config.get("auto_map"))
    spark_like = is_spark_architecture(config)
    plugin_ok = spark_plugin_installed() if spark_like else False
    if vllm_likely_supported(config, plugin_available=plugin_ok):
        report.engine_hint = "vllm"
        if plugin_ok:
            report.warnings.append(
                "Spark vLLM plugin detected in this interpreter; auto will use vLLM. "
                "If tool calls fail, pass `-- --enable-auto-tool-choice --tool-call-parser spark25`."
            )
    else:
        report.engine_hint = "hf"
        if spark_like:
            report.warnings.append(
                f"Architecture {report.architecture or report.model_type} needs the Spark vLLM plugin "
                "in THIS interpreter (`uv run` .venv). Installing it in AutoDL system/conda Python is ignored. "
                "Install into the project env, then rerun, or pass `--engine vllm`. Fallback: `--engine hf`."
            )
        else:
            report.warnings.append(
                f"Architecture {report.architecture or report.model_type or 'unknown'} is unlikely to work in vLLM. "
                "Use INFER_ENGINE=hf or --engine hf."
            )
    weight_gb, kv_gb, total_gb = estimate_memory_gb(
        config, metadata, max_model_len=max_model_len, dtype=dtype
    )
    report.estimated_weight_gb = None if weight_gb is None else round(weight_gb, 2)
    report.estimated_kv_gb = None if kv_gb is None else round(kv_gb, 2)
    report.estimated_total_gb = None if total_gb is None else round(total_gb, 2)
    available = gpu_memory_gb()
    report.gpu_memory_gb = None if available is None else round(available, 2)
    if report.estimated_total_gb and report.gpu_memory_gb:
        budget = report.gpu_memory_gb * gpu_memory_utilization
        if report.estimated_total_gb > budget:
            report.warnings.append(
                f"Estimated working set {report.estimated_total_gb:.2f} GiB exceeds "
                f"{gpu_memory_utilization:.0%} of GPU ({budget:.2f} GiB). "
                "Lower MAX_MODEL_LEN / GPU_MEMORY_UTILIZATION or use quantization."
            )
    if report.gpu_memory_gb is None:
        smi = nvidia_smi_memory_gb()
        if smi and not torch_cuda_available():
            report.warnings.append(
                f"nvidia-smi sees a GPU (~{smi:.1f} GiB) but this Python env has no CUDA torch. "
                "vLLM/HF will load on CPU. On AutoDL run `uv sync --extra infer` in this repo, "
                "and install Spark-plugin with `uv pip install` into `.venv`, not system Python."
            )
        else:
            report.warnings.append("No CUDA device visible; serving may fall back to CPU and be very slow.")
    report.ok = not report.problems
    LOGGER.info("Preflight: %s", json.dumps(report.as_dict(), ensure_ascii=False))
    return report
