from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from collections.abc import MutableMapping
from pathlib import Path

import psutil

from .bootstrap import SCRIPTS_DIR
from .log_follow import LogFollower
from .logging_utils import setup_logging
from .paths import ensure_dir
from .tee import REDACT_ENV

LOGGER = setup_logging("llm_tools.process")


def sanitize_omp_env(env: MutableMapping[str, str]) -> MutableMapping[str, str]:
    """Drop OMP_NUM_THREADS values libgomp rejects (empty, 0, non-integers)."""
    omp = env.get("OMP_NUM_THREADS")
    if omp is None:
        return env
    text = str(omp).strip()
    try:
        value = int(text)
    except ValueError:
        env.pop("OMP_NUM_THREADS", None)
        return env
    if value < 1:
        env.pop("OMP_NUM_THREADS", None)
    return env


def reset_log_file(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("", encoding="utf-8")


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
    *,
    tee_console: bool = True,
    secret: str | None = None,
) -> subprocess.Popen:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    reset_log_file(log_path)
    merged_env = dict(env) if env is not None else os.environ.copy()
    merged_env.setdefault("PYTHONUTF8", "1")
    merged_env.setdefault("PYTHONIOENCODING", "utf-8")
    merged_env.setdefault("PYTHONUNBUFFERED", "1")
    sanitize_omp_env(merged_env)
    if secret:
        merged_env[REDACT_ENV] = secret
    else:
        merged_env.pop(REDACT_ENV, None)

    tee_cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "run_tee.py"),
        "--log",
        str(log_path),
    ]
    if not tee_console:
        tee_cmd.append("--no-console")
    tee_cmd.extend(["--", *cmd])
    kwargs: dict = {
        "cwd": str(cwd),
        "env": merged_env,
    }
    if daemon:
        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kwargs["start_new_session"] = True
    LOGGER.info("Launching process cwd=%s (stdout/stderr teed to %s and console)", cwd, log_path)
    return subprocess.Popen(tee_cmd, **kwargs)


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


def wait_for_or_exit(
    proc: subprocess.Popen,
    predicate,
    timeout: float,
    interval: float = 1.0,
    description: str = "condition",
    log_path: Path | None = None,
    secret: str | None = None,
    follow_log: bool = False,
    log_start: int | None = None,
) -> str:
    """Wait until predicate() is true, the process exits, or timeout.

    Returns ``ok``, ``exited``, or ``timeout``.
    """
    follower = LogFollower(log_path, secret=secret, start_pos=log_start) if follow_log and log_path is not None else None
    if follower is not None:
        LOGGER.info("Streaming %s to console while waiting for health.", log_path)
    deadline = time.time() + timeout
    next_check = time.time()
    try:
        while time.time() < deadline:
            if follower is not None:
                follower.poll()
            now = time.time()
            if now >= next_check:
                if predicate():
                    return "ok"
                next_check = now + interval
            code = proc.poll()
            if code is not None:
                if follower is not None:
                    follower.poll()
                LOGGER.error("%s process exited with code %s before becoming ready.", description, code)
                return "exited"
            time.sleep(0.2)
        LOGGER.error("Timed out waiting for %s after %.1fs.", description, timeout)
        return "timeout"
    finally:
        if follower is not None:
            follower.close()


def current_python() -> str:
    return sys.executable
