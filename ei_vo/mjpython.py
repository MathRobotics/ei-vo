"""Helpers for launching MuJoCo viewers under ``mjpython`` on macOS."""

from __future__ import annotations

import os
import shutil
import sys
from collections.abc import Sequence

_MJPYTHON_ENV_VARS = ("MJPYTHON_BIN", "MJPYTHON_LIBPYTHON")


def is_running_under_mjpython() -> bool:
    """Return ``True`` when the current process was started by ``mjpython``."""

    return any(os.environ.get(name) for name in _MJPYTHON_ENV_VARS)


def maybe_relaunch_with_mjpython(renderer: str | None, *, exec_args: Sequence[str]) -> None:
    """Re-exec the current process via ``mjpython`` when MuJoCo needs it."""

    if renderer != "mujoco" or sys.platform != "darwin" or is_running_under_mjpython():
        return

    mjpython = shutil.which("mjpython")
    if mjpython is None:
        raise RuntimeError(
            "MuJoCo playback on macOS requires `mjpython`, but it was not found on PATH."
        )

    os.execvp(mjpython, [mjpython, *list(exec_args)])
