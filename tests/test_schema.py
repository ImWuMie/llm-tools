from __future__ import annotations

import pytest

from common.schema import SchemaError, raise_if_errors, validate_env_values, validate_train_config


def test_env_port_must_be_int() -> None:
    problems = validate_env_values({"VLLM_PORT": "abc"})
    assert any("VLLM_PORT" in item for item in problems)


def test_env_unknown_key_is_warning_unless_strict() -> None:
    values = {"MODEL_ID": "x", "TOTALLY_UNKNOWN": "1"}
    problems = validate_env_values(values, strict=False)
    assert any(item.startswith("warning:") for item in problems)
    strict = validate_env_values(values, strict=True)
    assert any("unknown .env keys" in item for item in strict)


def test_train_config_valid() -> None:
    payload = {"base_model": "./models/base", "output_dir": "./training/output"}
    assert validate_train_config(payload) == []


def test_train_config_report_to() -> None:
    payload = {"base_model": "./m", "output_dir": "./o", "report_to": "nope"}
    problems = validate_train_config(payload)
    assert any("report_to" in item for item in problems)


def test_raise_if_errors_ignores_warnings() -> None:
    raise_if_errors(["warning: x"], label="env")
    with pytest.raises(SchemaError):
        raise_if_errors(["VLLM_PORT: invalid"], label="env")
