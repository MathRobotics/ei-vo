"""Shared model-loading abstractions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ModelSource:
    """Normalized model path with an inferred format."""

    path: Path
    format: str

    @classmethod
    def from_value(cls, value: str | Path) -> "ModelSource":
        path = Path(value)
        suffix = path.suffix.lower()
        if suffix == ".urdf":
            return cls(path=path, format="urdf")
        return cls(path=path, format="mujoco")


__all__ = ["ModelSource"]
