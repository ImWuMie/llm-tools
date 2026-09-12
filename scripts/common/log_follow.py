from __future__ import annotations

import sys
from pathlib import Path
from typing import TextIO


class LogFollower:
    """Follow new bytes from a log file and write them to the console."""

    def __init__(
        self,
        path: Path,
        *,
        secret: str | None = None,
        stream: TextIO | None = None,
        start_pos: int | None = None,
    ) -> None:
        self.path = Path(path)
        self.secret = secret
        self.stream = stream or sys.stdout
        self._pos = start_pos if start_pos is not None else (self.path.stat().st_size if self.path.is_file() else 0)
        self._buf = ""

    def poll(self) -> None:
        if not self.path.is_file():
            return
        size = self.path.stat().st_size
        if size < self._pos:
            self._pos = 0
        with self.path.open("r", encoding="utf-8", errors="replace") as handle:
            handle.seek(self._pos)
            chunk = handle.read()
            self._pos = handle.tell()
        if not chunk:
            return
        if self.secret:
            chunk = chunk.replace(self.secret, "***")
        self._buf += chunk
        if "\n" not in self._buf:
            return
        lines = self._buf.split("\n")
        self._buf = lines[-1]
        text = "\n".join(lines[:-1]) + "\n"
        self.stream.write(text)
        self.stream.flush()

    def close(self) -> None:
        self.poll()
        if self._buf:
            line = self._buf
            if self.secret:
                line = line.replace(self.secret, "***")
            self.stream.write(line if line.endswith("\n") else line + "\n")
            self.stream.flush()
            self._buf = ""
