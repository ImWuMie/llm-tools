from __future__ import annotations

from typing import Any, Iterable, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator


class SchemaError(ValueError):
    """Invalid toolchain configuration."""


class EnvSettings(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True, str_strip_whitespace=True)

    MODEL_SOURCE: Literal["auto", "hf", "modelscope"] | None = None
    VLLM_PORT: int | None = None
    TENSOR_PARALLEL_SIZE: int | None = None
    MAX_MODEL_LEN: int | None = None
    VLLM_MAX_NUM_SEQS: int | None = None
    VLLM_HEALTH_TIMEOUT: float | None = None
    GPU_MEMORY_UTILIZATION: float | None = None
    VLLM_WINDOWS_BACKEND: Literal["auto", "wsl", "docker", "native", "fail"] | None = None
    INFER_ENGINE: Literal["auto", "vllm", "hf"] | None = None
    LICENSE_STRICT: bool | None = None
    CONFIG_STRICT: bool | None = None

    @field_validator("VLLM_PORT", "TENSOR_PARALLEL_SIZE", "MAX_MODEL_LEN", "VLLM_MAX_NUM_SEQS", mode="before")
    @classmethod
    def _empty_int(cls, value: Any) -> Any:
        if value == "" or value is None:
            return None
        return value

    @field_validator("VLLM_HEALTH_TIMEOUT", "GPU_MEMORY_UTILIZATION", mode="before")
    @classmethod
    def _empty_float(cls, value: Any) -> Any:
        if value == "" or value is None:
            return None
        return value

    @field_validator("LICENSE_STRICT", "CONFIG_STRICT", mode="before")
    @classmethod
    def _empty_bool(cls, value: Any) -> Any:
        if value == "" or value is None:
            return None
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"1", "true", "yes", "on", "y"}:
                return True
            if lowered in {"0", "false", "no", "off", "n"}:
                return False
        return value


KNOWN_ENV_PREFIXES = (
    "MODEL_",
    "HF_",
    "MODELSCOPE_",
    "DOWNLOAD_",
    "VLLM_",
    "TENSOR_",
    "GPU_",
    "MAX_MODEL_",
    "DTYPE",
    "QUANTIZATION",
    "CHECKPOINT_",
    "BASE_",
    "ADAPTER_",
    "TRAINED_",
    "MERGED_",
    "LOG_",
    "PID_",
    "INFER_",
    "LICENSE_",
    "CONFIG_",
    "EVAL_",
)


def _is_known_env_key(key: str) -> bool:
    if key in EnvSettings.model_fields:
        return True
    return any(key == prefix or key.startswith(prefix) for prefix in KNOWN_ENV_PREFIXES)


def validate_env_values(values: dict[str, str], *, strict: bool = False) -> list[str]:
    """Return human-readable problems. Unknown keys are warnings unless strict."""
    problems: list[str] = []
    payload = {key: value for key, value in values.items() if value != ""}
    try:
        EnvSettings.model_validate(payload)
    except ValidationError as exc:
        for error in exc.errors():
            loc = ".".join(str(item) for item in error.get("loc", ()))
            problems.append(f"{loc}: {error.get('msg')}")
    unknown = sorted(key for key in values if not _is_known_env_key(key))
    if unknown:
        message = "unknown .env keys: " + ", ".join(unknown)
        if strict:
            problems.append(message)
        else:
            problems.append("warning: " + message)
    return problems


class LoraSettings(BaseModel):
    model_config = ConfigDict(extra="allow")
    r: int = 8
    alpha: int = 16
    dropout: float = 0.05
    target_modules: list[str] = Field(default_factory=lambda: ["q_proj", "v_proj"])


class QuantizationSettings(BaseModel):
    model_config = ConfigDict(extra="allow")
    load_in_4bit: bool = False


class DataSettings(BaseModel):
    model_config = ConfigDict(extra="allow")
    skip_empty_lines: bool = True
    skip_comment_lines: bool = True
    comment_prefix: str = "#"
    encoding: str = "utf-8"


class TrainSettings(BaseModel):
    model_config = ConfigDict(extra="allow")
    base_model: str
    output_dir: str
    num_train_epochs: float = 3
    learning_rate: float = 2e-4
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    max_seq_length: int = 2048
    logging_steps: int = 10
    save_steps: int = 100
    save_total_limit: int = 3
    seed: int = 42
    fp16: bool = False
    bf16: bool = True
    gradient_checkpointing: bool = True
    report_to: str = "none"
    resume_from_checkpoint: str | None = None
    save_merged_model: bool = False
    lora: LoraSettings = Field(default_factory=LoraSettings)
    quantization: QuantizationSettings = Field(default_factory=QuantizationSettings)
    data: DataSettings = Field(default_factory=DataSettings)


def validate_train_config(payload: dict[str, Any], *, strict: bool = False) -> list[str]:
    problems: list[str] = []
    try:
        TrainSettings.model_validate(payload)
    except ValidationError as exc:
        for error in exc.errors():
            loc = ".".join(str(item) for item in error.get("loc", ()))
            problems.append(f"{loc}: {error.get('msg')}")
    if strict:
        allowed = set(TrainSettings.model_fields)
        extra = sorted(key for key in payload if key not in allowed)
        if extra:
            problems.append("unknown training config keys: " + ", ".join(extra))
    report_to = str(payload.get("report_to") or "none").strip().lower()
    if report_to not in {"none", "tensorboard", "wandb", "all"}:
        problems.append("report_to must be none / tensorboard / wandb / all")
    return problems


def raise_if_errors(problems: Iterable[str], *, label: str) -> None:
    errors = [item for item in problems if not item.startswith("warning:")]
    if errors:
        pretty = "; ".join(errors)
        raise SchemaError(f"{label} is invalid: {pretty}")
