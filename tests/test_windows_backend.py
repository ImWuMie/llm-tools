from __future__ import annotations

import pytest

from common.platform_utils import resolve_windows_vllm_backend
from common.vllm_launcher import VLLMLaunchError, cli_windows_backend


def test_force_native_wins() -> None:
    assert (
        resolve_windows_vllm_backend(
            "wsl",
            force_native=True,
            vllm_importable=False,
            wsl_available=True,
            docker_available=True,
        )
        == "native"
    )


def test_auto_prefers_importable_vllm() -> None:
    assert (
        resolve_windows_vllm_backend(
            "auto",
            vllm_importable=True,
            wsl_available=True,
            docker_available=True,
        )
        == "native"
    )


def test_auto_falls_back_to_wsl_then_docker() -> None:
    assert (
        resolve_windows_vllm_backend(
            "auto",
            vllm_importable=False,
            wsl_available=True,
            docker_available=True,
        )
        == "wsl"
    )
    assert (
        resolve_windows_vllm_backend(
            "auto",
            vllm_importable=False,
            wsl_available=False,
            docker_available=True,
        )
        == "docker"
    )
    assert (
        resolve_windows_vllm_backend(
            "auto",
            vllm_importable=False,
            wsl_available=False,
            docker_available=False,
        )
        == "fail"
    )


def test_default_requested_is_wsl() -> None:
    assert (
        resolve_windows_vllm_backend(
            None,
            vllm_importable=True,
            wsl_available=True,
            docker_available=True,
        )
        == "wsl"
    )


def test_unknown_backend_raises() -> None:
    with pytest.raises(ValueError, match="Unknown VLLM_WINDOWS_BACKEND"):
        resolve_windows_vllm_backend("hypervisor")


def test_cli_windows_backend_exclusive() -> None:
    assert cli_windows_backend(native=True) == ("native", True)
    assert cli_windows_backend(force_native=True) == ("native", True)
    assert cli_windows_backend(wsl=True) == ("wsl", False)
    assert cli_windows_backend(docker=True) == ("docker", False)
    with pytest.raises(VLLMLaunchError, match="Conflicting"):
        cli_windows_backend(wsl=True, native=True)
