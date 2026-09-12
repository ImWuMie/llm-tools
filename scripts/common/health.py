from __future__ import annotations

from typing import Any

import requests

from .logging_utils import setup_logging
from .secrets import mask_secret

LOGGER = setup_logging("llm_tools.health")


def models_endpoint(host: str, port: int) -> str:
    check_host = "127.0.0.1" if host in {"0.0.0.0", "::", ""} else host
    return f"http://{check_host}:{int(port)}/v1/models"


def check_openai_models(host: str, port: int, api_key: str | None = None, timeout: float = 5.0) -> dict[str, Any]:
    url = models_endpoint(host, port)
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    LOGGER.debug("Health check GET %s api_key=%s", url, mask_secret(api_key) if api_key else "<empty>")
    response = requests.get(url, headers=headers, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    names = [item.get("id") for item in payload.get("data", []) if isinstance(item, dict)]
    LOGGER.info("Health check OK. models=%s", names)
    return payload
