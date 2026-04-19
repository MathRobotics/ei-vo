"""Model inspection helpers independent from rendering backends."""

from __future__ import annotations

from pathlib import Path

from .common import ModelSource
from .mujoco import (
    ArmJointMap,
    clamp_to_limits,
    clip_positions_to_limits,
    detect_arm_joint_qaddr,
    detect_arm_joints,
    load_mujoco_model,
    load_mujoco_robot_model,
)
from .urdf import load_urdf_robot_model


def load_robot_model(
    model_path: str | Path,
    expected_dof: int | None = None,
):
    """Load robot metadata using the lightest available loader for the format."""

    source = ModelSource.from_value(model_path)
    if source.format == "urdf":
        return load_urdf_robot_model(source.path, expected_dof=expected_dof)
    return load_mujoco_robot_model(source.path, expected_dof=expected_dof)


__all__ = [
    "ArmJointMap",
    "ModelSource",
    "clamp_to_limits",
    "clip_positions_to_limits",
    "detect_arm_joint_qaddr",
    "detect_arm_joints",
    "load_mujoco_model",
    "load_mujoco_robot_model",
    "load_robot_model",
    "load_urdf_robot_model",
]
