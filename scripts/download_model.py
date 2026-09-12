#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from common.bootstrap import PROJECT_ROOT, ensure_sys_path

ensure_sys_path()

from common.env import ConfigError, apply_runtime_env, load_app_config, upsert_env_key
from common.logging_utils import setup_logging
from common.secrets import mask_secret
from common.validate_model import is_valid_local_model, validate_local_model
from common.license_check import inspect_license
from model_sources import DownloadRequest, SourceError, get_source, resolve_source_order

LOGGER = setup_logging("llm_tools.download")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a model from Hugging Face and/or ModelScope.")
    parser.add_argument("--source", choices=["auto", "hf", "modelscope"], default=None)
    parser.add_argument(
        "--model-id",
        default=None,
        help=(
            "Override MODEL_ID for both sources. Takes priority over HF_MODEL_ID / "
            "MODELSCOPE_MODEL_ID from .env unless --hf-model-id / --modelscope-model-id is set."
        ),
    )
    parser.add_argument("--hf-model-id", default=None)
    parser.add_argument("--modelscope-model-id", default=None)
    parser.add_argument("--revision", default=None, help="Fallback revision if source-specific revision is empty.")
    parser.add_argument("--hf-revision", default=None)
    parser.add_argument("--modelscope-revision", default=None)
    parser.add_argument("--output-dir", default=None, help="Download root, equivalent to DOWNLOAD_DIR.")
    parser.add_argument("--output-name", default=None)
    parser.add_argument("--token", default=None, help="Optional token override for the selected source.")
    parser.add_argument("--force", action="store_true", help="Re-download even if a valid local model exists.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--update-env", action="store_true", help="Write MODEL_DIR back to .env on success.")
    parser.add_argument("--skip-license-check", action="store_true")
    return parser.parse_args()


def pick(cli_value: str | None, *env_values: str | None) -> str | None:
    if cli_value:
        return cli_value
    for value in env_values:
        if value:
            return value
    return None


def model_id_for(source: str, args: argparse.Namespace, config) -> str:
    # CLI source-specific > CLI --model-id > env source-specific > MODEL_ID
    if source == "hf":
        value = pick(
            args.hf_model_id,
            args.model_id,
            config.get("HF_MODEL_ID"),
            config.get("MODEL_ID"),
        )
    else:
        value = pick(
            args.modelscope_model_id,
            args.model_id,
            config.get("MODELSCOPE_MODEL_ID"),
            config.get("MODEL_ID"),
        )
    if not value:
        raise ConfigError(
            f"No model id configured for source `{source}`. "
            "Set MODEL_ID, or HF_MODEL_ID / MODELSCOPE_MODEL_ID."
        )
    return value


def revision_for(source: str, args: argparse.Namespace, config) -> str:
    if source == "hf":
        return pick(args.hf_revision, args.revision, config.get("HF_REVISION"), "main") or "main"
    return pick(args.modelscope_revision, args.revision, config.get("MODELSCOPE_REVISION"), "master") or "master"


def token_for(source: str, args: argparse.Namespace, config) -> str | None:
    if args.token:
        return args.token
    if source == "hf":
        return config.get("HF_TOKEN") or None
    return config.get("MODELSCOPE_API_TOKEN") or None


def endpoint_for(source: str, config) -> str | None:
    if source == "hf":
        return config.get("HF_ENDPOINT")
    return config.get("MODELSCOPE_ENDPOINT")


def resolve_source_choice(args: argparse.Namespace, config) -> tuple[str, str]:
    """CLI --source wins over .env MODEL_SOURCE. Returns (source, origin)."""
    cli = (args.source or "").strip().lower()
    env_source = (config.get("MODEL_SOURCE") or "").strip().lower()
    env_path = getattr(config, "env_path", ".env")
    if cli:
        if cli == "auto" and env_source not in {"", "auto"}:
            LOGGER.warning(
                "CLI --source auto overrides .env MODEL_SOURCE=%s from %s. "
                "Omit --source to honor .env, or pass --source %s.",
                env_source,
                env_path,
                env_source,
            )
        return cli, "cli --source"
    if env_source:
        return env_source, f".env MODEL_SOURCE ({env_path})"
    return "auto", "default"


def build_output_dir(args: argparse.Namespace, config) -> Path:
    download_root = Path(args.output_dir) if args.output_dir else config.get_path("DOWNLOAD_DIR")
    if download_root is None:
        download_root = PROJECT_ROOT / "models"
    if not download_root.is_absolute():
        download_root = PROJECT_ROOT / download_root
    cli_id = pick(args.model_id, args.hf_model_id, args.modelscope_model_id)
    if args.output_name:
        name = args.output_name
    elif cli_id:
        # A CLI model id should not reuse the .env folder name (often a different model).
        name = Path(cli_id).name
    else:
        fallback_id = pick(
            config.get("MODEL_OUTPUT_NAME"),
            config.get("MODEL_ID"),
            config.get("HF_MODEL_ID"),
            config.get("MODELSCOPE_MODEL_ID"),
        )
        name = pick(config.get("MODEL_OUTPUT_NAME"), Path(fallback_id).name if fallback_id else None)
    if not name:
        raise ConfigError("MODEL_OUTPUT_NAME is required so Hugging Face and ModelScope share one local path.")
    return download_root.resolve() / name


def maybe_skip(output_dir: Path, force: bool) -> bool:
    if force:
        return False
    if is_valid_local_model(output_dir):
        LOGGER.info("Valid local model already exists at %s. Skipping download (use --force to re-download).", output_dir)
        return True
    if output_dir.exists():
        problems = validate_local_model(output_dir)
        LOGGER.warning("Existing directory %s is incomplete: %s", output_dir, "; ".join(problems))
    return False


def download_one(source_name: str, args: argparse.Namespace, config, output_dir: Path, dry_run: bool) -> Path:
    request = DownloadRequest(
        source=source_name,
        model_id=model_id_for(source_name, args, config),
        revision=revision_for(source_name, args, config),
        output_dir=output_dir,
        token=token_for(source_name, args, config),
        endpoint=endpoint_for(source_name, config),
        extra={"hf_transfer": config.get("HF_HUB_ENABLE_HF_TRANSFER", "0")},
    )
    LOGGER.info(
        "Trying source=%s model_id=%s revision=%s token=%s output=%s",
        source_name,
        request.model_id,
        request.revision,
        mask_secret(request.token),
        request.output_dir,
    )
    if dry_run:
        LOGGER.info("Dry-run: would download via %s", source_name)
        return output_dir
    source = get_source(source_name)
    return source.download(request)


def main() -> int:
    args = parse_args()
    try:
        config = load_app_config()
        apply_runtime_env(config)
        source, source_origin = resolve_source_choice(args, config)
        order = resolve_source_order(source, config.get("MODEL_SOURCE_PRIORITY"))
        output_dir = build_output_dir(args, config)
        LOGGER.info("Loaded %s", config.env_path)
        LOGGER.info(
            "source=%s (from %s) order=%s output_dir=%s",
            source,
            source_origin,
            order,
            output_dir,
        )
        if source == "auto" and order[:1] == ["hf"]:
            LOGGER.info(
                "auto tries Hugging Face first. A hang is not a failure, so ModelScope "
                "will not start until HF errors. On AutoDL / mainland networks use "
                "`--source modelscope` or set MODEL_SOURCE_PRIORITY=modelscope,hf."
            )

        if maybe_skip(output_dir, args.force):
            if not args.skip_license_check:
                report = inspect_license(output_dir)
                for warning in report.warnings:
                    LOGGER.warning("%s", warning)
                if config.get_bool("LICENSE_STRICT", False) and report.commercial_ok is False:
                    LOGGER.error("LICENSE_STRICT=1 and the model license looks restricted.")
                    return 1
            print(output_dir)
            return 0
        if args.force and output_dir.exists() and not args.dry_run:
            LOGGER.warning("Removing existing directory because --force was set: %s", output_dir)
            shutil.rmtree(output_dir)

        failures: list[str] = []
        for name in order:
            try:
                downloaded = download_one(name, args, config, output_dir, args.dry_run)
                if args.dry_run:
                    LOGGER.info("Dry-run complete for %s", name)
                    print(downloaded)
                    return 0
                problems = validate_local_model(downloaded)
                if problems:
                    raise SourceError(f"Download from {name} completed but validation failed: {'; '.join(problems)}")
                LOGGER.info("Download succeeded via %s. Local path: %s", name, downloaded)
                if not args.skip_license_check:
                    license_report = inspect_license(downloaded)
                    for warning in license_report.warnings:
                        LOGGER.warning("%s", warning)
                    if config.get_bool("LICENSE_STRICT", False) and license_report.commercial_ok is False:
                        raise SourceError(
                            "LICENSE_STRICT=1 and the downloaded model license looks non-commercial or restricted. "
                            "Review the model card or rerun with --skip-license-check."
                        )
                LOGGER.info("Point MODEL_DIR at this path if you want vLLM to load it by default.")
                if args.update_env:
                    rel = Path(downloaded)
                    try:
                        rel = Path(downloaded).resolve().relative_to(PROJECT_ROOT)
                        value = "./" + str(rel).replace("\\", "/")
                    except ValueError:
                        value = str(downloaded)
                    upsert_env_key(config.env_path, "MODEL_DIR", value)
                    LOGGER.info("Updated MODEL_DIR in %s (value written, tokens untouched).", config.env_path)
                print(downloaded)
                return 0
            except Exception as exc:
                failures.append(f"{name}: {exc}")
                LOGGER.warning("Source %s failed: %s", name, exc)

        LOGGER.error("All sources failed:")
        for item in failures:
            LOGGER.error("  - %s", item)
        LOGGER.error(
            "Actionable next steps:\n"
            "  1) Check network / mirror settings (HF_ENDPOINT, MODELSCOPE_ENDPOINT).\n"
            "  2) If the repo is gated, set HF_TOKEN or MODELSCOPE_API_TOKEN in .env.\n"
            "  3) Confirm the model ids exist: HF_MODEL_ID / MODELSCOPE_MODEL_ID.\n"
            "  4) Retry a single source: --source hf  or  --source modelscope."
        )
        return 1
    except (ConfigError, SourceError) as exc:
        LOGGER.error("%s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
