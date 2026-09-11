from __future__ import annotations

from typing import Any, Mapping

SECRET_KEYS = {
    "HF_TOKEN",
    "MODELSCOPE_API_TOKEN",
    "VLLM_API_KEY",
    "TOKEN",
    "API_KEY",
    "ACCESS_TOKEN",
}


def mask_secret(value: str | None) -> str:
    if not value:
        return ""
    text = str(value)
    if len(text) <= 8:
        return "***"
    return f"{text[:4]}***{text[-2:]}"


def is_secret_key(key: str) -> bool:
    upper = key.upper()
    if upper in SECRET_KEYS:
        return True
    return any(token in upper for token in ("TOKEN", "API_KEY", "SECRET", "PASSWORD"))


def redact_mapping(data: Mapping[str, Any]) -> dict[str, Any]:
    redacted: dict[str, Any] = {}
    for key, value in data.items():
        if is_secret_key(str(key)) and value:
            redacted[key] = mask_secret(str(value))
        else:
            redacted[key] = value
    return redacted


def redact_command(cmd: list[str]) -> list[str]:
    hidden_flags = {"--api-key", "--token", "--hf-token", "--modelscope-token"}
    out: list[str] = []
    hide_next = False
    for item in cmd:
        if hide_next:
            out.append("***")
            hide_next = False
            continue
        if item in hidden_flags:
            out.append(item)
            hide_next = True
            continue
        if "=" in item:
            flag, value = item.split("=", 1)
            if flag in hidden_flags or is_secret_key(flag.lstrip("-")):
                out.append(f"{flag}=***")
                continue
        out.append(item)
    return out
