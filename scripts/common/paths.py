from __future__ import annotations

from pathlib import Path

from .bootstrap import PROJECT_ROOT


def resolve_path(value: str | Path | None, base: Path | None = None) -> Path | None:
    if value is None or str(value).strip() == "":
        return None
    path = Path(str(value).strip()).expanduser()
    if not path.is_absolute():
        path = (base or PROJECT_ROOT) / path
    return path.resolve()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def display_path(path: Path | str) -> str:
    return str(Path(path))


def to_wsl_path(path: Path) -> str:
    resolved = path.resolve()
    drive = resolved.drive
    if len(drive) >= 2 and drive[1] == ":":
        rest = str(resolved)[len(drive) :].replace("\\", "/")
        return f"/mnt/{drive[0].lower()}{rest}"
    return str(resolved).replace("\\", "/")
