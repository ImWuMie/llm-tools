#!/usr/bin/env python3
from __future__ import annotations

import argparse
import compileall
import json
import sys

from common.bootstrap import PROJECT_ROOT, SCRIPTS_DIR, ensure_sys_path

ensure_sys_path()

from common.data_format import DataOptions, parse_training_file
from common.env import load_app_config
from common.logging_utils import setup_logging
from common.platform_utils import (
    cuda_visible,
    has_docker,
    has_wsl,
    is_windows,
    platform_name,
    probe_vllm_import,
    resolve_windows_vllm_backend,
    vllm_native_supported,
)
from common.process import port_in_use
from model_sources import resolve_source_order

LOGGER = setup_logging("llm_tools.selfcheck")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline/online toolchain self-check.")
    parser.add_argument("--offline", action="store_true", help="Skip GPU/network-sensitive checks.")
    return parser.parse_args()


def check(title: str, ok: bool, detail: str) -> dict:
    status = "ok" if ok else "fail"
    LOGGER.info("[%s] %s: %s", status.upper(), title, detail)
    return {"name": title, "ok": ok, "detail": detail}


def main() -> int:
    args = parse_args()
    results: list[dict] = []
    results.append(check("python", sys.version_info >= (3, 10), sys.version.replace("\n", " ")))

    compile_ok = compileall.compile_dir(str(SCRIPTS_DIR), quiet=1, force=False)
    results.append(check("compileall scripts", bool(compile_ok), str(SCRIPTS_DIR)))

    env_ok = True
    detail = ""
    try:
        config = load_app_config()
        detail = str(config.env_path)
        order = resolve_source_order(config.get("MODEL_SOURCE") or "auto", config.get("MODEL_SOURCE_PRIORITY"))
        results.append(check("source order", bool(order), ",".join(order)))
        host = config.get("VLLM_HOST") or "0.0.0.0"
        port = int(config.get("VLLM_PORT") or 8000)
        busy = port_in_use(host, port)
        results.append(check("port config", True, f"{host}:{port} in_use={busy}"))
        model_dir = config.get_path("MODEL_DIR")
        results.append(check("paths", model_dir is not None, f"MODEL_DIR={model_dir}"))
    except Exception as exc:
        env_ok = False
        detail = str(exc)
        config = None
    results.append(check("env", env_ok, detail))

    sample = PROJECT_ROOT / "training" / "data" / "sample.txt"
    try:
        samples, report = parse_training_file(sample, "default", DataOptions())
        results.append(check("sample data", report.samples >= 1, json.dumps(report.as_dict(), ensure_ascii=False)))
        results.append(check("sample roles", samples[0][0]["role"] in {"system", "user"}, str(samples[0][0])))
    except Exception as exc:
        results.append(check("sample data", False, str(exc)))

    results.append(check("platform", True, platform_name()))
    results.append(check("vllm native", True, f"supported={vllm_native_supported()} wsl={has_wsl()} docker={has_docker()}"))
    importable, detail = probe_vllm_import()
    results.append(check("vllm import", True, f"importable={importable} {detail}"))
    if is_windows() and config is not None:
        requested = config.get("VLLM_WINDOWS_BACKEND") or "wsl"
        resolved = resolve_windows_vllm_backend(
            requested,
            vllm_importable=importable,
            wsl_available=has_wsl(),
            docker_available=has_docker(),
        )
        results.append(
            check(
                "windows vllm backend",
                True,
                f"requested={requested} resolved={resolved}",
            )
        )
    if not args.offline:
        results.append(check("cuda", True, f"visible={cuda_visible()}"))

    failed = [item for item in results if not item["ok"]]
    print(json.dumps({"failed": len(failed), "results": results}, ensure_ascii=False, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
