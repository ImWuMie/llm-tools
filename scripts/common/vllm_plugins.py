from __future__ import annotations

import importlib.metadata
import importlib.util
import os

from .logging_utils import setup_logging

LOGGER = setup_logging("llm_tools.vllm_plugins")

SPARK_MODEL_TYPES = {"spark2_5", "spark25", "spark"}
SPARK_ARCHITECTURES = {"Spark2_5ForCausalLM", "Spark25ForCausalLM"}
SPARK_DIST_NAMES = {
    "spark-plugin",
    "vllm-spark",
    "spark-vllm",
    "vllm-spark-plugin",
    "spark2-5-plugin",
}
SPARK_MODULE_CANDIDATES = (
    "spark_plugin",
    "vllm_spark",
    "spark_vllm",
    "spark.plugin",
)


def _norm(name: str) -> str:
    return name.lower().replace("_", "-").strip()


def _truthy(value: str | None) -> bool | None:
    if value is None:
        return None
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "on", "y"}:
        return True
    if lowered in {"0", "false", "no", "off", "n"}:
        return False
    return None


def installed_distribution_names() -> set[str]:
    names: set[str] = set()
    try:
        for dist in importlib.metadata.distributions():
            raw = dist.metadata.get("Name") if dist.metadata else None
            if raw:
                names.add(_norm(raw))
    except Exception:
        return names
    return names


def vllm_plugin_entry_text() -> str:
    chunks: list[str] = []
    try:
        eps = importlib.metadata.entry_points()
    except Exception:
        return ""
    groups = ("vllm.general_plugins", "vllm.plugins", "vllm.platform_plugins")
    for group in groups:
        try:
            selected = eps.select(group=group) if hasattr(eps, "select") else eps.get(group, [])  # type: ignore[arg-type]
        except Exception:
            continue
        for ep in selected or []:
            chunks.append(f"{getattr(ep, 'name', '')}={getattr(ep, 'value', '')}")
    return " ".join(chunks).lower()


def spark_plugin_installed() -> bool:
    """True if the Spark vLLM plugin is importable in *this* interpreter.

    Installing the plugin in AutoDL's system/conda env does not count for `uv run`.
    """
    forced = _truthy(os.environ.get("VLLM_SPARK_PLUGIN"))
    if forced is not None:
        return forced
    names = installed_distribution_names()
    if names & SPARK_DIST_NAMES:
        return True
    if any("spark" in name and "plugin" in name for name in names):
        return True
    if "spark" in vllm_plugin_entry_text():
        return True
    for mod in SPARK_MODULE_CANDIDATES:
        try:
            if importlib.util.find_spec(mod) is not None:
                return True
        except Exception:
            continue
    return False


def is_spark_architecture(config: dict) -> bool:
    model_type = str(config.get("model_type") or "").strip().lower()
    if model_type in SPARK_MODEL_TYPES:
        return True
    architectures = config.get("architectures") or []
    return any(str(item) in SPARK_ARCHITECTURES for item in architectures)


def plugin_covers_architecture(config: dict, *, plugin_available: bool | None = None) -> bool:
    if not is_spark_architecture(config):
        return False
    available = spark_plugin_installed() if plugin_available is None else plugin_available
    return bool(available)
