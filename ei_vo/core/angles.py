"""Angle loading helpers shared across demos and utilities."""

from __future__ import annotations

import json
import pathlib
from typing import Iterable

import numpy as np


def load_angles(path: str | pathlib.Path, deg: bool) -> np.ndarray:
    """Load a ``(T, DOF)`` array of joint angles from ``CSV``/``NPY``/``JSON``.

    Parameters
    ----------
    path:
        Path to the input file. Supported extensions are ``.csv``, ``.npy``,
        and ``.json`` (case-insensitive).
    deg:
        Whether the provided angles are in degrees. When ``True`` the data is
        converted to radians before returning.

    Returns
    -------
    np.ndarray
        A ``float`` array with shape ``(T, DOF)``.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If the file extension is unsupported or the loaded array is not 2-D.
    """

    p = pathlib.Path(path)
    if not p.exists():
        raise FileNotFoundError(path)

    suffix = p.suffix.lower()
    if suffix == ".csv":
        arr = np.loadtxt(p, delimiter=",", dtype=float)
    elif suffix == ".npy":
        arr = np.load(p)
    elif suffix == ".json":
        with p.open("r", encoding="utf-8") as f:
            data: Iterable[Iterable[float]] = json.load(f)
        arr = np.array(data, dtype=float)
    else:
        raise ValueError(f"Unsupported file extension: {p.suffix}")

    if arr.ndim != 2:
        raise ValueError(f"angles must be a 2D array. Got {arr.shape}")

    if deg:
        arr = np.deg2rad(arr)

    return arr
