from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"


def ensure_sys_path() -> None:
    root = str(PROJECT_ROOT)
    scripts = str(SCRIPTS_DIR)
    if root not in sys.path:
        sys.path.insert(0, root)
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
