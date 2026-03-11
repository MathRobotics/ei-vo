"""Core data structures used across the package."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


def _as_float_array(name: str, value: np.ndarray | list[float], *, ndim: int) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != ndim:
        raise ValueError(f"{name} must be a {ndim}D array. Got {array.shape}.")
    return array


@dataclass(slots=True)
class Trajectory:
    """Validated trajectory container."""

    q: np.ndarray
    t: np.ndarray | None = None
    dq: np.ndarray | None = None
    ddq: np.ndarray | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.q = _as_float_array("q", self.q, ndim=2)

        if self.t is not None:
            self.t = _as_float_array("t", self.t, ndim=1)
            if self.t.shape[0] != self.q.shape[0]:
                raise ValueError(
                    f"t must have {self.q.shape[0]} samples to match q. Got {self.t.shape[0]}."
                )

        if self.dq is not None:
            self.dq = _as_float_array("dq", self.dq, ndim=2)
            if self.dq.shape != self.q.shape:
                raise ValueError(f"dq must match q shape {self.q.shape}. Got {self.dq.shape}.")

        if self.ddq is not None:
            self.ddq = _as_float_array("ddq", self.ddq, ndim=2)
            if self.ddq.shape != self.q.shape:
                raise ValueError(
                    f"ddq must match q shape {self.q.shape}. Got {self.ddq.shape}."
                )

        self.meta = dict(self.meta)

    @property
    def steps(self) -> int:
        return int(self.q.shape[0])

    @property
    def dof(self) -> int:
        return int(self.q.shape[1])

    @classmethod
    def from_positions(
        cls,
        positions: np.ndarray | list[list[float]] | list[float],
        *,
        dt: float | None = None,
        meta: dict[str, Any] | None = None,
    ) -> "Trajectory":
        q = np.asarray(positions, dtype=float)
        if q.ndim == 1:
            q = q[None, :]
        if q.ndim != 2:
            raise ValueError(f"positions must be a 1D or 2D array. Got {q.shape}.")

        t = None
        if dt is not None:
            if dt <= 0:
                raise ValueError(f"dt must be positive. Got {dt}.")
            t = np.arange(q.shape[0], dtype=float) * dt
        return cls(q=q, t=t, meta=meta or {})

    @classmethod
    def coerce(cls, value: "Trajectory" | np.ndarray | list[list[float]] | list[float]) -> "Trajectory":
        if isinstance(value, cls):
            return value
        return cls.from_positions(value)


@dataclass(slots=True)
class RobotModel:
    """Minimal robot description extracted from an MJCF model."""

    name: str
    joint_names: tuple[str, ...] | list[str]
    limits: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.joint_names = tuple(self.joint_names)
        if self.limits is not None:
            self.limits = _as_float_array("limits", self.limits, ndim=2)
            expected_shape = (len(self.joint_names), 2)
            if self.limits.shape != expected_shape:
                raise ValueError(
                    f"limits must have shape {expected_shape}. Got {self.limits.shape}."
                )

    @property
    def dof(self) -> int:
        return len(self.joint_names)

    def clamp(self, q: np.ndarray) -> np.ndarray:
        positions = _as_float_array("q", q, ndim=2).copy()
        if positions.shape[1] != self.dof:
            raise ValueError(
                f"q must have {self.dof} columns to match the model. Got {positions.shape[1]}."
            )
        if self.limits is None:
            return positions
        for index, (lower, upper) in enumerate(self.limits):
            if lower < upper:
                positions[:, index] = np.clip(positions[:, index], lower, upper)
        return positions


__all__ = ["RobotModel", "Trajectory"]
