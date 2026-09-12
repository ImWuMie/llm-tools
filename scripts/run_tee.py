#!/usr/bin/env python3
from __future__ import annotations

import sys

from common.bootstrap import ensure_sys_path

ensure_sys_path()

from common.tee import main

if __name__ == "__main__":
    sys.exit(main())
