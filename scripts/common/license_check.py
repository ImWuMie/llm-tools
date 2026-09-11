from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .logging_utils import setup_logging

LOGGER = setup_logging("llm_tools.license")

RESTRICTIVE_HINTS = (
    "non-commercial",
    "noncommercial",
    "research only",
    "research-only",
    "no commercial",
    "cc-by-nc",
    "cc by-nc",
    "llama 2 community",
    "llama 3 community",
)

PERMISSIVE = {"apache-2.0", "apache 2.0", "mit", "bsd-2-clause", "bsd-3-clause", "apache2"}


@dataclass
class LicenseReport:
    model_dir: str
    license: str | None = None
    source_file: str | None = None
    commercial_ok: bool | None = None
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_dir": self.model_dir,
            "license": self.license,
            "source_file": self.source_file,
            "commercial_ok": self.commercial_ok,
            "warnings": self.warnings,
        }


def _front_matter(text: str) -> dict[str, str]:
    if not text.startswith("---"):
        return {}
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}
    data: dict[str, str] = {}
    for line in parts[1].splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        data[key.strip().lower()] = value.strip().strip("\"'")
    return data


def inspect_license(model_dir: Path) -> LicenseReport:
    report = LicenseReport(model_dir=str(model_dir))
    candidates = [
        model_dir / "README.md",
        model_dir / "LICENSE",
        model_dir / "license",
        model_dir / "LICENSE.txt",
        model_dir / "configuration.json",
    ]
    for path in candidates:
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if path.name.lower().startswith("readme"):
            meta = _front_matter(text)
            if meta.get("license"):
                report.license = meta["license"]
                report.source_file = path.name
                break
        if path.name.lower().startswith("license"):
            first = " ".join(text.splitlines()[:8])
            match = re.search(r"(apache(?:-|\s*)2\.0|mit|bsd|llama|cc-by[^\s,]*)", first, re.I)
            report.license = match.group(1) if match else first[:80].strip() or "see LICENSE"
            report.source_file = path.name
            break
        if path.name == "configuration.json":
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict) and payload.get("license"):
                report.license = str(payload["license"])
                report.source_file = path.name
                break
    if not report.license:
        report.warnings.append("No license file or model-card license field was found.")
        report.commercial_ok = None
        LOGGER.warning("License check: unknown for %s", model_dir)
        return report

    lowered = report.license.lower()
    if any(hint in lowered for hint in RESTRICTIVE_HINTS) or "nc" in lowered.split():
        report.commercial_ok = False
        report.warnings.append(
            f"License `{report.license}` looks non-commercial or restricted. Review before production use."
        )
    elif lowered in PERMISSIVE or "apache" in lowered or lowered == "mit":
        report.commercial_ok = True
    else:
        report.commercial_ok = None
        report.warnings.append(
            f"License `{report.license}` was found but commercial use is not auto-classified. Read the model card."
        )
    LOGGER.info("License check: %s", json.dumps(report.as_dict(), ensure_ascii=False))
    return report
