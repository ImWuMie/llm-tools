from __future__ import annotations

import os
from pathlib import Path

from common.logging_utils import setup_logging
from common.secrets import mask_secret

from .base import DownloadRequest, ModelSource, SourceError, call_supported

LOGGER = setup_logging("llm_tools.source.hf")


class HuggingFaceSource(ModelSource):
    name = "hf"

    def download(self, request: DownloadRequest) -> Path:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise SourceError(
                "huggingface_hub is not installed. Run `uv sync` or `uv sync --extra download`."
            ) from exc

        output_dir = Path(request.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if request.endpoint:
            os.environ["HF_ENDPOINT"] = request.endpoint
        extra = request.extra or {}
        enable_transfer = str(extra.get("hf_transfer", os.environ.get("HF_HUB_ENABLE_HF_TRANSFER", "0")))
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = enable_transfer

        LOGGER.info(
            "Downloading from Hugging Face repo=%s revision=%s endpoint=%s token=%s -> %s",
            request.model_id,
            request.revision or "main",
            request.endpoint or os.environ.get("HF_ENDPOINT", "https://huggingface.co"),
            mask_secret(request.token),
            output_dir,
        )
        try:
            call_supported(
                snapshot_download,
                repo_id=request.model_id,
                revision=request.revision or "main",
                local_dir=str(output_dir),
                token=request.token or None,
                endpoint=request.endpoint,
                resume_download=True,
            )
        except Exception as exc:
            raise SourceError(f"Hugging Face download failed for {request.model_id}: {exc}") from exc
        return output_dir
