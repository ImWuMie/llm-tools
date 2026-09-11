from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import psutil

from .logging_utils import setup_logging
from .paths import ensure_dir

LOGGER = setup_logging("llm_tools.process")


def pid_file(pid_dir: Path, name: str) -> Path:
    return ensure_dir(pid_dir) / f"{name}.pid"


def log_file(log_dir: Path, name: str) -> Path:
    return ensure_dir(log_dir) / f"{name}.log"


def read_pid(path: Path) -> int | None:
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def write_pid(path: Path, pid: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(pid) + "\n", encoding="utf-8", newline="\n")


def is_pid_running(pid: int) -> bool:
    try:
        return psutil.pid_exists(pid) and psutil.Process(pid).is_running()
    except psutil.Error:
        return False


def port_in_use(host: str, port: int) -> bool:
    check_host = "127.0.0.1" if host in {"0.0.0.0", "::", ""} else host
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1)
        return sock.connect_ex((check_host, int(port))) == 0


def start_process(
    cmd: list[str],
    cwd: Path,
    log_path: Path,
    env: dict[str, str] | None = None,
    daemon: bool = False,
) -> subprocess.Popen:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    merged_env.setdefault("PYTHONUTF8", "1")
    merged_env.setdefault("PYTHONIOENCODING", "utf-8")

    stdout = None if not daemon else open(log_path, "a", encoding="utf-8")  # noqa: SIM115
    stderr = None if not daemon else subprocess.STDOUT
    kwargs: dict = {
        "cwd": str(cwd),
        "env": merged_env,
    }
    if daemon:
        kwargs["stdout"] = stdout
        kwargs["stderr"] = stderr
        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | getattr(
                subprocess, "DETACHED_PROCESS", 0
            )
        else:
            kwargs["start_new_session"] = True
    LOGGER.info("Launching process cwd=%s", cwd)
    return subprocess.Popen(cmd, **kwargs)


def stop_pid(pid: int, timeout: float = 20.0) -> None:
    if not is_pid_running(pid):
        LOGGER.info("PID %s is not running.", pid)
        return
    proc = psutil.Process(pid)
    children = proc.children(recursive=True)
    targets = children + [proc]
    LOGGER.info("Stopping PID %s and %s child process(es).", pid, len(children))
    for item in targets:
        try:
            item.terminate()
        except psutil.Error:
            continue
    gone, alive = psutil.wait_procs(targets, timeout=timeout)
    LOGGER.info("Terminated %s process(es).", len(gone))
    for item in alive:
        try:
            LOGGER.warning("Force killing PID %s.", item.pid)
            item.kill()
        except psutil.Error:
            continue


def wait_for(predicate, timeout: float, interval: float = 1.0, description: str = "condition") -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    LOGGER.error("Timed out waiting for %s after %.1fs.", description, timeout)
    return False


def current_python() -> str:
    return sys.executable
