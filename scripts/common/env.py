from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from dotenv import dotenv_values, load_dotenv

from .bootstrap import PROJECT_ROOT
from .logging_utils import setup_logging
from .paths import ensure_dir, resolve_path
from .secrets import redact_mapping

LOGGER = setup_logging("llm_tools.env")

TRUE_VALUES = {"1", "true", "yes", "on", "y"}
FALSE_VALUES = {"0", "false", "no", "off", "n", ""}


class ConfigError(RuntimeError):
    """Raised when required configuration is missing or invalid."""


@dataclass(frozen=True)
class AppConfig:
    values: dict[str, str]
    env_path: Path
    project_root: Path

    def get(self, key: str, default: str | None = None) -> str | None:
        value = self.values.get(key)
        if value is None or value == "":
            return default
        return value

    def require(self, key: str) -> str:
        value = self.get(key)
        if value is None:
            raise ConfigError(
                f"Missing required config `{key}`. "
                f"Set it in {self.env_path} or pass a CLI override."
            )
        return value

    def get_bool(self, key: str, default: bool = False) -> bool:
        value = self.get(key)
        if value is None:
            return default
        lowered = value.strip().lower()
        if lowered in TRUE_VALUES:
            return True
        if lowered in FALSE_VALUES:
            return False
        raise ConfigError(f"Invalid boolean for `{key}`: {value}")

    def get_int(self, key: str, default: int | None = None) -> int | None:
        value = self.get(key)
        if value is None:
            return default
        try:
            return int(value)
        except ValueError as exc:
            raise ConfigError(f"Invalid integer for `{key}`: {value}") from exc

    def get_float(self, key: str, default: float | None = None) -> float | None:
        value = self.get(key)
        if value is None:
            return default
        try:
            return float(value)
        except ValueError as exc:
            raise ConfigError(f"Invalid float for `{key}`: {value}") from exc

    def get_path(self, key: str, default: str | None = None) -> Path | None:
        value = self.get(key, default)
        return resolve_path(value, self.project_root)

    def require_path(self, key: str) -> Path:
        path = self.get_path(key)
        if path is None:
            raise ConfigError(f"Missing required path `{key}`.")
        return path

    def require_keys(self, keys: Iterable[str]) -> None:
        missing = [key for key in keys if not self.get(key)]
        if missing:
            pretty = ", ".join(missing)
            raise ConfigError(
                f"Missing required variables: {pretty}. "
                f"Edit {self.env_path} (see .env_example) and retry."
            )

    def redacted(self) -> dict[str, str]:
        return redact_mapping(self.values)

    def override(self, **kwargs: str | None) -> None:
        for key, value in kwargs.items():
            if value is not None and str(value) != "":
                self.values[key] = str(value)


def env_example_path(project_root: Path | None = None) -> Path:
    return (project_root or PROJECT_ROOT) / ".env_example"


def env_path(project_root: Path | None = None) -> Path:
    return (project_root or PROJECT_ROOT) / ".env"


def ensure_env_file(project_root: Path | None = None, copy_if_missing: bool = True) -> Path:
    root = project_root or PROJECT_ROOT
    target = env_path(root)
    example = env_example_path(root)
    if target.exists():
        return target
    if not example.exists():
        raise ConfigError(
            f".env not found at {target} and .env_example is also missing. "
            "Create .env from the documented template before running scripts."
        )
    if not copy_if_missing:
        raise ConfigError(
            f".env not found at {target}. Copy {example} to .env, fill in tokens/paths, then retry."
        )
    shutil.copyfile(example, target)
    LOGGER.warning("Created %s from .env_example. Review values before downloading or serving models.", target)
    return target


def load_app_config(project_root: Path | None = None, copy_if_missing: bool = True) -> AppConfig:
    root = project_root or PROJECT_ROOT
    path = ensure_env_file(root, copy_if_missing=copy_if_missing)
    load_dotenv(path, override=False)
    raw = dotenv_values(path)
    values: dict[str, str] = {}
    for key, value in raw.items():
        if key is None:
            continue
        if value is None:
            values[key] = os.environ.get(key, "")
        else:
            values[key] = str(value)
            os.environ.setdefault(key, str(value))
    return AppConfig(values=values, env_path=path, project_root=root)


def upsert_env_key(path: Path, key: str, value: str) -> None:
    lines: list[str] = []
    if path.exists():
        lines = path.read_text(encoding="utf-8").splitlines()
    updated = False
    new_lines: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in line:
            new_lines.append(line)
            continue
        current_key, _current_value = line.split("=", 1)
        if current_key.strip() == key:
            new_lines.append(f"{key}={value}")
            updated = True
        else:
            new_lines.append(line)
    if not updated:
        if new_lines and new_lines[-1].strip():
            new_lines.append("")
        new_lines.append(f"{key}={value}")
    path.write_text("\n".join(new_lines) + "\n", encoding="utf-8", newline="\n")


def apply_runtime_env(config: AppConfig) -> None:
    """Propagate non-secret runtime paths and Hub endpoints into os.environ."""
    hf_endpoint = config.get("HF_ENDPOINT")
    if hf_endpoint:
        os.environ["HF_ENDPOINT"] = hf_endpoint
    hf_transfer = config.get("HF_HUB_ENABLE_HF_TRANSFER", "0")
    if hf_transfer is not None:
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = hf_transfer
    ms_endpoint = config.get("MODELSCOPE_ENDPOINT")
    if ms_endpoint:
        os.environ["MODELSCOPE_DOMAIN"] = ms_endpoint
        os.environ["MODELSCOPE_ENDPOINT"] = ms_endpoint
    ensure_dir(config.require_path("LOG_DIR") if config.get("LOG_DIR") else config.project_root / "logs")
    ensure_dir(config.require_path("PID_DIR") if config.get("PID_DIR") else config.project_root / "run")
