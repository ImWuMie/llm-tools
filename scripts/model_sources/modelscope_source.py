from __future__ import annotations

import inspect
import os
import shutil
from pathlib import Path

from common.logging_utils import setup_logging
from common.secrets import mask_secret

from .base import DownloadRequest, ModelSource, SourceError, call_supported

LOGGER = setup_logging("llm_tools.source.modelscope")


def _snapshot_download_fn():
    try:
        from modelscope import snapshot_download

        return snapshot_download
    except Exception:
        try:
            from modelscope.hub.snapshot_download import snapshot_download

            return snapshot_download
        except Exception as exc:
            raise SourceError(
                "modelscope is not installed or failed to import. "
                "Run `uv sync --extra download`. If it conflicts with torch/vLLM, "
                "use a dedicated download environment."
            ) from exc


def _copy_if_needed(downloaded: Path, output_dir: Path) -> Path:
    downloaded = Path(downloaded)
    output_dir = Path(output_dir)
    if downloaded.resolve() == output_dir.resolve():
        return output_dir
    if output_dir.exists() and any(output_dir.iterdir()):
        return output_dir
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if downloaded.is_dir():
        shutil.copytree(downloaded, output_dir, dirs_exist_ok=True)
        return output_dir
    raise SourceError(f"ModelScope returned a non-directory path: {downloaded}")


class ModelScopeSource(ModelSource):
    name = "modelscope"

    def download(self, request: DownloadRequest) -> Path:
        snapshot_download = _snapshot_download_fn()
        output_dir = Path(request.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if request.token:
            os.environ.setdefault("MODELSCOPE_API_TOKEN", request.token)
        if request.endpoint:
            os.environ["MODELSCOPE_DOMAIN"] = request.endpoint
            os.environ["MODELSCOPE_ENDPOINT"] = request.endpoint

        params = inspect.signature(snapshot_download).parameters
        kwargs: dict = {}
        if "model_id" in params:
            kwargs["model_id"] = request.model_id
        elif "repo_id" in params:
            kwargs["repo_id"] = request.model_id
        else:
            kwargs["model_id"] = request.model_id

        if "revision" in params:
            kwargs["revision"] = request.revision or "master"
        if "local_dir" in params:
            kwargs["local_dir"] = str(output_dir)
        elif "cache_dir" in params:
            kwargs["cache_dir"] = str(output_dir.parent)

        token_keys = ("token", "access_token", "user_access_token")
        for key in token_keys:
            if request.token and key in params:
                kwargs[key] = request.token
                break
        if request.endpoint and "endpoint" in params:
            kwargs["endpoint"] = request.endpoint

        LOGGER.info(
            "Downloading from ModelScope model_id=%s revision=%s endpoint=%s token=%s -> %s",
            request.model_id,
            request.revision or "master",
            request.endpoint or os.environ.get("MODELSCOPE_ENDPOINT") or "<default>",
            mask_secret(request.token),
            output_dir,
        )
        try:
            result = call_supported(snapshot_download, **kwargs)
        except TypeError:
            # Older signatures accept positional model_id only.
            result = snapshot_download(request.model_id, revision=request.revision or "master")
        except Exception as exc:
            raise SourceError(f"ModelScope download failed for {request.model_id}: {exc}") from exc

        downloaded = Path(result) if result else output_dir
        return _copy_if_needed(downloaded, output_dir)
