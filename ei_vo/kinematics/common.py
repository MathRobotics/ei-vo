"""Common data structures for kinematics backends."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..core import Trajectory


@dataclass(slots=True)
class KinematicsResult:
    """Batch forward-kinematics result."""

    transforms: np.ndarray
    backend: str
    model_path: str | None = None
    base_link: str | None = None
    end_link: str | None = None
    meta: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        self.transforms = np.asarray(self.transforms, dtype=float)
        if self.transforms.ndim != 3 or self.transforms.shape[1:] != (4, 4):
            raise ValueError(
                f"transforms must have shape (T, 4, 4). Got {self.transforms.shape}."
            )
        self.model_path = None if self.model_path is None else Path(self.model_path).as_posix()
        self.meta = dict(self.meta or {})

    @property
    def steps(self) -> int:
        return int(self.transforms.shape[0])

    @property
    def positions(self) -> np.ndarray:
        return self.transforms[:, :3, 3]


def coerce_trajectory(value) -> Trajectory:
    """Normalize trajectory-like input to :class:`Trajectory`."""

    return Trajectory.coerce(value)


def make_transform(
    rotation: np.ndarray | None = None,
    translation: np.ndarray | None = None,
) -> np.ndarray:
    """Build a homogeneous transform from rotation and translation."""

    transform = np.eye(4, dtype=float)
    if rotation is not None:
        transform[:3, :3] = np.asarray(rotation, dtype=float).reshape(3, 3)
    if translation is not None:
        transform[:3, 3] = np.asarray(translation, dtype=float).reshape(3)
    return transform


__all__ = ["KinematicsResult", "coerce_trajectory", "make_transform"]
