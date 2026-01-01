"""Recording-related helpers shared across demos and utilities."""

from __future__ import annotations

import os
import pathlib
import time
from typing import Optional, Tuple


def resolve_record_destination(
    record_arg: Optional[os.PathLike | str],
    *,
    prefix: str = "demo_",
    suffix: str = ".mp4",
) -> Tuple[Optional[str], Optional[str]]:
    """Resolve a recording destination from a CLI-style ``--record`` argument.

    Parameters
    ----------
    record_arg:
        Value passed to ``--record``. ``None`` means "no recording". An empty
        string or a path ending with ``os.sep`` indicates that the caller wants
        to automatically generate a filename under the given directory (or the
        default ``./recordings`` directory when empty).
    prefix:
        Filename prefix to use when auto-generating a file.
    suffix:
        Filename suffix/extension to use when auto-generating a file.

    Returns
    -------
    tuple[str | None, str | None]
        A pair ``(record_path, auto_dir)``. ``record_path`` is ``None`` when no
        recording was requested. ``auto_dir`` is the directory that should be
        created by the caller when an auto-generated filename is used; it is
        ``None`` when the caller provided a concrete filename.
    """

    if record_arg is None:
        return None, None

    record_value = os.fspath(record_arg).strip()

    if record_value == "":
        base_dir = pathlib.Path.cwd() / "recordings"
    else:
        candidate = pathlib.Path(record_value)
        if record_value.endswith(os.sep):
            base_dir = candidate
        elif candidate.exists() and candidate.is_dir():
            base_dir = candidate
        else:
            return candidate.as_posix(), None

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"{prefix}{timestamp}{suffix}"
    record_path = (base_dir / filename).as_posix()
    return record_path, base_dir.as_posix()

