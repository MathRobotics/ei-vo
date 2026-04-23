"""Pure-Python URDF metadata and scene loading."""

from __future__ import annotations

import pathlib
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np

from ..core import RobotModel

_ACTUATED_JOINT_TYPES = {"continuous", "prismatic", "revolute"}
_SUPPORTED_JOINT_TYPES = _ACTUATED_JOINT_TYPES | {"fixed"}
_DEFAULT_RGBA = np.array([0.7, 0.7, 0.7, 1.0], dtype=float)


@dataclass(frozen=True, slots=True)
class UrdfJoint:
    name: str
    joint_type: str
    parent_link: str
    child_link: str
    origin: np.ndarray
    axis: np.ndarray


@dataclass(frozen=True, slots=True)
class UrdfVisual:
    geometry_type: str
    origin: np.ndarray
    rgba: np.ndarray
    size: np.ndarray | None = None
    radius: float | None = None
    length: float | None = None


@dataclass(slots=True)
class UrdfScene:
    name: str
    root_link: str
    joint_names: tuple[str, ...]
    limits: np.ndarray
    child_joints: dict[str, tuple[UrdfJoint, ...]]
    link_visuals: dict[str, tuple[UrdfVisual, ...]]

    @property
    def dof(self) -> int:
        return len(self.joint_names)

    def clamp(self, q: np.ndarray) -> np.ndarray:
        positions = np.asarray(q, dtype=float).copy()
        if positions.ndim != 2:
            raise ValueError(f"Trajectory positions must be a 2D array. Got {positions.shape}.")
        if positions.shape[1] != self.dof:
            raise ValueError(
                f"Trajectory dof ({positions.shape[1]}) does not match scene dof ({self.dof})."
            )
        for index, (lower, upper) in enumerate(self.limits):
            if lower < upper:
                positions[:, index] = np.clip(positions[:, index], lower, upper)
        return positions


def _joint_sort_key(name: str) -> int:
    match = re.search(r"(\d+)$", name) or re.search(r"joint[_-]?(\d+)", name)
    return int(match.group(1)) if match else 999


def _parse_space_separated_floats(
    value: str | None,
    *,
    size: int,
    default: tuple[float, ...],
) -> np.ndarray:
    if value is None:
        return np.asarray(default, dtype=float)
    pieces = [piece for piece in value.replace(",", " ").split() if piece]
    if len(pieces) != size:
        raise ValueError(f"Expected {size} floats, got {value!r}.")
    return np.asarray([float(piece) for piece in pieces], dtype=float)


def _rotation_matrix_from_rpy(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = [float(value) for value in rpy]
    cr = float(np.cos(roll))
    sr = float(np.sin(roll))
    cp = float(np.cos(pitch))
    sp = float(np.sin(pitch))
    cy = float(np.cos(yaw))
    sy = float(np.sin(yaw))
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=float,
    )


def _transform_from_xyz_rpy(xyz: np.ndarray, rpy: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = _rotation_matrix_from_rpy(rpy)
    transform[:3, 3] = np.asarray(xyz, dtype=float)
    return transform


def _origin_transform(element: ET.Element | None) -> np.ndarray:
    if element is None:
        return np.eye(4, dtype=float)
    return _transform_from_xyz_rpy(
        _parse_space_separated_floats(element.get("xyz"), size=3, default=(0.0, 0.0, 0.0)),
        _parse_space_separated_floats(element.get("rpy"), size=3, default=(0.0, 0.0, 0.0)),
    )


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


def _is_arm_joint(joint: ET.Element) -> bool:
    name = (joint.get("name") or "").strip()
    if not name:
        return False
    lowered = name.lower()
    return "finger" not in lowered and "gripper" not in lowered


def _parse_material_rgba(
    material: ET.Element | None,
    *,
    named_materials: dict[str, np.ndarray],
) -> np.ndarray:
    if material is None:
        return _DEFAULT_RGBA.copy()

    color = material.find("color")
    if color is not None and color.get("rgba"):
        rgba = _parse_space_separated_floats(color.get("rgba"), size=4, default=tuple(_DEFAULT_RGBA))
        return np.clip(rgba, 0.0, 1.0)

    name = (material.get("name") or "").strip()
    if name and name in named_materials:
        return named_materials[name].copy()
    return _DEFAULT_RGBA.copy()


def _parse_visual(
    visual: ET.Element,
    *,
    named_materials: dict[str, np.ndarray],
) -> UrdfVisual | None:
    geometry = visual.find("geometry")
    if geometry is None:
        return None

    origin = _origin_transform(visual.find("origin"))
    rgba = _parse_material_rgba(visual.find("material"), named_materials=named_materials)

    box = geometry.find("box")
    if box is not None and box.get("size"):
        return UrdfVisual(
            geometry_type="box",
            origin=origin,
            rgba=rgba,
            size=_parse_space_separated_floats(box.get("size"), size=3, default=(0.1, 0.1, 0.1)),
        )

    cylinder = geometry.find("cylinder")
    if cylinder is not None and cylinder.get("radius") and cylinder.get("length"):
        return UrdfVisual(
            geometry_type="cylinder",
            origin=origin,
            rgba=rgba,
            radius=float(cylinder.get("radius")),
            length=float(cylinder.get("length")),
        )

    sphere = geometry.find("sphere")
    if sphere is not None and sphere.get("radius"):
        return UrdfVisual(
            geometry_type="sphere",
            origin=origin,
            rgba=rgba,
            radius=float(sphere.get("radius")),
        )

    return None


def _parse_joint(joint: ET.Element) -> UrdfJoint | None:
    joint_type = (joint.get("type") or "").strip().lower()
    if joint_type not in _SUPPORTED_JOINT_TYPES:
        return None

    name = (joint.get("name") or "").strip()
    if not name:
        return None

    parent = joint.find("parent")
    child = joint.find("child")
    parent_link = (parent.get("link") or "").strip() if parent is not None else ""
    child_link = (child.get("link") or "").strip() if child is not None else ""
    if not parent_link or not child_link:
        return None

    axis = _parse_space_separated_floats(
        joint.find("axis").get("xyz") if joint.find("axis") is not None else None,
        size=3,
        default=(0.0, 0.0, 1.0),
    )
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm > 1e-12:
        axis = axis / axis_norm
    else:
        axis = np.array([0.0, 0.0, 1.0], dtype=float)

    return UrdfJoint(
        name=name,
        joint_type=joint_type,
        parent_link=parent_link,
        child_link=child_link,
        origin=_origin_transform(joint.find("origin")),
        axis=axis,
    )


def _axis_angle_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    ax = np.asarray(axis, dtype=float)
    norm = float(np.linalg.norm(ax))
    if norm <= 1e-12 or abs(float(angle)) <= 1e-12:
        return np.eye(3, dtype=float)
    x, y, z = ax / norm
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    t = 1.0 - c
    return np.array(
        [
            [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
            [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
            [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
        ],
        dtype=float,
    )


def _joint_motion_transform(joint: UrdfJoint, value: float) -> np.ndarray:
    transform = np.eye(4, dtype=float)
    if joint.joint_type in {"revolute", "continuous"}:
        transform[:3, :3] = _axis_angle_rotation(joint.axis, value)
    elif joint.joint_type == "prismatic":
        transform[:3, 3] = np.asarray(joint.axis, dtype=float) * float(value)
    return transform


def load_urdf_scene(
    model_path: str | pathlib.Path,
    expected_dof: int | None = None,
) -> UrdfScene:
    """Load a lightweight URDF scene graph for pure-Python rendering."""

    path = pathlib.Path(model_path)
    tree = ET.parse(path)
    root = tree.getroot()

    named_materials: dict[str, np.ndarray] = {}
    for material in root.findall("material"):
        name = (material.get("name") or "").strip()
        if not name:
            continue
        named_materials[name] = _parse_material_rgba(material, named_materials={})

    child_joints: dict[str, list[UrdfJoint]] = {}
    child_links: set[str] = set()
    actuated: list[tuple[UrdfJoint, np.ndarray]] = []
    for joint_elem in root.findall("joint"):
        joint = _parse_joint(joint_elem)
        if joint is None:
            continue
        child_joints.setdefault(joint.parent_link, []).append(joint)
        child_links.add(joint.child_link)
        if joint.joint_type in _ACTUATED_JOINT_TYPES and _is_arm_joint(joint_elem):
            actuated.append((joint, _joint_limits(joint_elem)))

    actuated.sort(key=lambda item: _joint_sort_key(item[0].name))
    if expected_dof is not None:
        if len(actuated) < expected_dof:
            raise RuntimeError(
                f"Model provides {len(actuated)} arm joints, but {expected_dof} were requested."
            )
        actuated = actuated[:expected_dof]

    joint_names = tuple(joint.name for joint, _limits in actuated)
    if not joint_names:
        raise RuntimeError("No actuated arm joints found in URDF model.")

    limits = np.vstack([joint_limits for _joint, joint_limits in actuated])
    link_names = [
        (link.get("name") or "").strip()
        for link in root.findall("link")
        if (link.get("name") or "").strip()
    ]
    root_candidates = [link_name for link_name in link_names if link_name not in child_links]
    if not root_candidates:
        raise RuntimeError("Could not determine the root link for the URDF model.")

    link_visuals: dict[str, tuple[UrdfVisual, ...]] = {}
    for link in root.findall("link"):
        link_name = (link.get("name") or "").strip()
        if not link_name:
            continue
        visuals = tuple(
            visual
            for visual in (
                _parse_visual(visual_elem, named_materials=named_materials)
                for visual_elem in link.findall("visual")
            )
            if visual is not None
        )
        link_visuals[link_name] = visuals

    return UrdfScene(
        name=(root.get("name") or path.stem).strip() or path.stem,
        root_link=root_candidates[0],
        joint_names=joint_names,
        limits=limits,
        child_joints={name: tuple(joints) for name, joints in child_joints.items()},
        link_visuals=link_visuals,
    )


def load_urdf_robot_model(
    model_path: str | pathlib.Path,
    expected_dof: int | None = None,
) -> RobotModel:
    """Load arm-joint metadata from a URDF without optional backends."""

    scene = load_urdf_scene(model_path, expected_dof=expected_dof)
    return RobotModel(
        name=scene.name,
        joint_names=scene.joint_names,
        limits=scene.limits,
    )


def compute_link_poses(
    scene: UrdfScene,
    joint_values: np.ndarray | list[float],
) -> dict[str, np.ndarray]:
    """Compute link-frame poses for a single URDF configuration."""

    row = np.asarray(joint_values, dtype=float).reshape(-1)
    if row.shape[0] != scene.dof:
        raise ValueError(f"Expected {scene.dof} joint values, got {row.shape[0]}.")
    config = {joint_name: float(value) for joint_name, value in zip(scene.joint_names, row)}

    poses: dict[str, np.ndarray] = {scene.root_link: np.eye(4, dtype=float)}
    stack = [scene.root_link]
    while stack:
        parent_link = stack.pop()
        parent_pose = poses[parent_link]
        for joint in scene.child_joints.get(parent_link, ()):
            motion = _joint_motion_transform(joint, config.get(joint.name, 0.0))
            child_pose = parent_pose @ joint.origin @ motion
            poses[joint.child_link] = child_pose
            stack.append(joint.child_link)
    return poses


__all__ = [
    "UrdfJoint",
    "UrdfScene",
    "UrdfVisual",
    "compute_link_poses",
    "load_urdf_robot_model",
    "load_urdf_scene",
]
