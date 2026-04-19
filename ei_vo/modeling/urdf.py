"""Pure-Python URDF metadata loading."""

from __future__ import annotations

import pathlib
import re
import xml.etree.ElementTree as ET

import numpy as np

from ..core import RobotModel

_SUPPORTED_JOINT_TYPES = {"continuous", "prismatic", "revolute"}


def _joint_sort_key(name: str) -> int:
    match = re.search(r"(\d+)$", name) or re.search(r"joint[_-]?(\d+)", name)
    return int(match.group(1)) if match else 999


def _iter_arm_joints(root: ET.Element):
    joints = []
    for joint in root.findall("joint"):
        joint_type = (joint.get("type") or "").strip().lower()
        if joint_type not in _SUPPORTED_JOINT_TYPES:
            continue
        name = (joint.get("name") or "").strip()
        if not name:
            continue
        lowered = name.lower()
        if "finger" in lowered or "gripper" in lowered:
            continue
        joints.append(joint)
    joints.sort(key=lambda joint: _joint_sort_key((joint.get("name") or "").strip()))
    return joints


def _joint_limits(joint: ET.Element) -> np.ndarray:
    joint_type = (joint.get("type") or "").strip().lower()
    if joint_type == "continuous":
        return np.array([-np.inf, np.inf], dtype=float)

    limit = joint.find("limit")
    if limit is None:
        return np.array([-np.inf, np.inf], dtype=float)

    lower = limit.get("lower")
    upper = limit.get("upper")
    if lower is None or upper is None:
        return np.array([-np.inf, np.inf], dtype=float)
    return np.array([float(lower), float(upper)], dtype=float)


def load_urdf_robot_model(
    model_path: str | pathlib.Path,
    expected_dof: int | None = None,
) -> RobotModel:
    """Load arm-joint metadata from a URDF without optional backends."""

    path = pathlib.Path(model_path)
    tree = ET.parse(path)
    root = tree.getroot()
    joints = _iter_arm_joints(root)

    if expected_dof is not None:
        if len(joints) < expected_dof:
            raise RuntimeError(
                f"Model provides {len(joints)} arm joints, but {expected_dof} were requested."
            )
        joints = joints[:expected_dof]

    joint_names = tuple((joint.get("name") or "").strip() for joint in joints)
    if not joint_names:
        raise RuntimeError("No actuated arm joints found in URDF model.")

    limits = np.vstack([_joint_limits(joint) for joint in joints])
    return RobotModel(
        name=(root.get("name") or path.stem).strip() or path.stem,
        joint_names=joint_names,
        limits=limits,
    )


__all__ = ["load_urdf_robot_model"]
