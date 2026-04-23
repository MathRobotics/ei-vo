"""Model inspection helpers independent from rendering backends."""

from __future__ import annotations

from pathlib import Path

from .common import ModelSource
from .urdf import (
    UrdfJoint,
    UrdfScene,
    UrdfVisual,
    compute_link_poses,
    load_urdf_robot_model,
    load_urdf_scene,
)


def load_robot_model(
    model_path: str | Path,
    expected_dof: int | None = None,
):
    """Load robot metadata using the built-in URDF parser."""

    source = ModelSource.from_value(model_path)
    return load_urdf_robot_model(source.path, expected_dof=expected_dof)


__all__ = [
    "ModelSource",
    "UrdfJoint",
    "UrdfScene",
    "UrdfVisual",
    "compute_link_poses",
    "load_robot_model",
    "load_urdf_robot_model",
    "load_urdf_scene",
]
