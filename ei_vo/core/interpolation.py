"""Interpolation helpers shared across demos and utilities."""

from __future__ import annotations

import numpy as np


def quintic(q0: np.ndarray, q1: np.ndarray, T: float, dt: float) -> np.ndarray:
    """Scalar quintic polynomial with zero velocity/acceleration at both ends.

    Parameters
    ----------
    q0, q1:
        Start and end positions (arrays of the same shape).
    T:
        Duration of the segment in seconds.
    dt:
        Sampling period in seconds.

    Returns
    -------
    np.ndarray
        Array of interpolated positions including both endpoints. The shape is
        ``(ceil(T/dt)+1, *q0.shape)``.
    """

    if T <= 0:
        raise ValueError(f"T must be positive. Got {T}.")
    if dt <= 0:
        raise ValueError(f"dt must be positive. Got {dt}.")

    q0 = np.asarray(q0, dtype=float)
    q1 = np.asarray(q1, dtype=float)
    if q0.shape != q1.shape:
        raise ValueError(f"q0 and q1 must have the same shape. Got {q0.shape} and {q1.shape}.")

    t = np.arange(0.0, T + 1e-12, dt)
    s = t / max(T, 1e-9)
    a = 10 * s**3 - 15 * s**4 + 6 * s**5
    return q0[None, ...] + (q1 - q0)[None, ...] * a[..., None]
