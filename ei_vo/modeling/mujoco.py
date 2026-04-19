"""MuJoCo-backed model inspection helpers."""

from __future__ import annotations

import contextlib
import pathlib
import re
import shutil
import tempfile
import urllib.parse
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np

from ..core import RobotModel


def _import_mujoco():
    try:
        import mujoco as mj
    except ImportError as exc:
        raise RuntimeError("MuJoCo support requires the optional dependency 'mujoco'.") from exc
    return mj


@dataclass(slots=True)
class ArmJointMap:
    """Resolved arm joint metadata for a MuJoCo model."""

    joint_ids: tuple[int, ...]
    joint_names: tuple[str, ...]
    qpos_addresses: tuple[int, ...]
    limits: np.ndarray

    @property
    def dof(self) -> int:
        return len(self.qpos_addresses)


def _joint_sort_key(name: str) -> int:
    match = re.search(r"(\d+)$", name) or re.search(r"joint[_-]?(\d+)", name)
    return int(match.group(1)) if match else 999


def detect_arm_joints(model, expected_dof: int | None = None) -> ArmJointMap:
    """Collect arm hinge joints while skipping grippers and fingers."""

    mj = _import_mujoco()
    joint_ids: list[int] = []
    joint_names: list[str] = []
    qpos_addresses: list[int] = []

    for joint_id in range(model.njnt):
        if model.jnt_type[joint_id] != mj.mjtJoint.mjJNT_HINGE:
            continue
        joint_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, joint_id) or ""
        lowered = joint_name.lower()
        if "finger" in lowered or "gripper" in lowered:
            continue
        joint_ids.append(joint_id)
        joint_names.append(joint_name)
        qpos_addresses.append(int(model.jnt_qposadr[joint_id]))

    if not joint_ids:
        raise RuntimeError("No arm hinge joints found in model.")

    order = np.argsort([_joint_sort_key(name) for name in joint_names])
    joint_ids = [joint_ids[index] for index in order]
    joint_names = [joint_names[index] for index in order]
    qpos_addresses = [qpos_addresses[index] for index in order]

    if expected_dof is not None:
        if len(joint_ids) < expected_dof:
            raise RuntimeError(
                f"Model provides {len(joint_ids)} arm joints, but {expected_dof} were requested."
            )
        joint_ids = joint_ids[:expected_dof]
        joint_names = joint_names[:expected_dof]
        qpos_addresses = qpos_addresses[:expected_dof]

    limits = np.asarray([model.jnt_range[joint_id] for joint_id in joint_ids], dtype=float)
    return ArmJointMap(
        joint_ids=tuple(joint_ids),
        joint_names=tuple(joint_names),
        qpos_addresses=tuple(qpos_addresses),
        limits=limits,
    )


def detect_arm_joint_qaddr(model, expected_dof: int | None = None) -> list[int]:
    """Compatibility helper returning only qpos addresses."""

    return list(detect_arm_joints(model, expected_dof=expected_dof).qpos_addresses)


def _resolve_urdf_mesh_source(model_dir: pathlib.Path, filename: str) -> pathlib.Path | None:
    parsed = urllib.parse.urlparse(filename)
    if parsed.scheme in {"", "file"}:
        raw_path = urllib.parse.unquote(parsed.path if parsed.scheme == "file" else filename)
        candidate = pathlib.Path(raw_path)
        if not candidate.is_absolute():
            candidate = model_dir / candidate
        return candidate if candidate.is_file() else None

    if parsed.scheme == "package":
        tail = pathlib.Path(urllib.parse.unquote(parsed.path.lstrip("/")))
        package_name = urllib.parse.unquote(parsed.netloc)
        search_roots = (model_dir, *model_dir.parents)
        for root in search_roots:
            for candidate in (root / tail, root / package_name / tail):
                if candidate.is_file():
                    return candidate
        return None

    return None


@contextlib.contextmanager
def _prepared_mujoco_model_path(model_path: str | pathlib.Path):
    path = pathlib.Path(model_path)
    if path.suffix.lower() != ".urdf":
        yield path
        return

    tree = ET.parse(path)
    root = tree.getroot()
    staged_assets: list[tuple[pathlib.Path, str]] = []

    # MuJoCo's URDF importer flattens mesh filenames to their basenames when
    # resolving mesh assets, so nested relative paths need to be staged.
    for index, mesh in enumerate(root.findall(".//mesh[@filename]")):
        filename = mesh.get("filename")
        if filename is None:
            continue
        source = _resolve_urdf_mesh_source(path.parent, filename)
        if source is None:
            continue
        staged_name = f"asset_{index:04d}{source.suffix}"
        mesh.set("filename", staged_name)
        staged_assets.append((source, staged_name))

    if not staged_assets:
        yield path
        return

    with tempfile.TemporaryDirectory(prefix="ei_vo_mj_urdf_") as tmp_dir:
        staged_dir = pathlib.Path(tmp_dir)
        staged_path = staged_dir / path.name
        tree.write(staged_path, encoding="utf-8", xml_declaration=True)
        for source, staged_name in staged_assets:
            shutil.copy2(source, staged_dir / staged_name)
        yield staged_path


def load_mujoco_model(model_path: str | pathlib.Path):
    """Load a MuJoCo model, staging URDF assets when necessary."""

    mj = _import_mujoco()
    with _prepared_mujoco_model_path(model_path) as prepared_path:
        return mj.MjModel.from_xml_path(prepared_path.as_posix())


def load_mujoco_robot_model(
    model_path: str | pathlib.Path,
    expected_dof: int | None = None,
) -> RobotModel:
    """Load arm-joint metadata from a MuJoCo-readable model."""

    path = pathlib.Path(model_path)
    model = load_mujoco_model(path)
    arm_joints = detect_arm_joints(model, expected_dof=expected_dof)
    return RobotModel(
        name=path.stem,
        joint_names=arm_joints.joint_names,
        limits=arm_joints.limits,
    )


def clip_positions_to_limits(q: np.ndarray, limits: np.ndarray) -> np.ndarray:
    """Clamp trajectory positions to per-joint limits."""

    positions = np.asarray(q, dtype=float).copy()
    if positions.ndim != 2:
        raise ValueError(f"Trajectory positions must be 2D. Got {positions.shape}.")
    if positions.shape[1] != limits.shape[0]:
        raise ValueError(
            f"Trajectory dof ({positions.shape[1]}) does not match detected joints ({limits.shape[0]})."
        )

    for index, (lower, upper) in enumerate(limits):
        if lower < upper:
            positions[:, index] = np.clip(positions[:, index], lower, upper)
    return positions


def clamp_to_limits(model, arm_qaddr: list[int], q: np.ndarray) -> np.ndarray:
    """Compatibility helper to clamp positions to model limits."""

    joint_limits = []
    for qpos_address in arm_qaddr:
        matches = np.flatnonzero(model.jnt_qposadr == qpos_address)
        if matches.size == 0:
            joint_limits.append(np.array([-np.inf, np.inf], dtype=float))
            continue
        joint_limits.append(np.asarray(model.jnt_range[int(matches[0])], dtype=float))
    return clip_positions_to_limits(q, np.asarray(joint_limits, dtype=float))


__all__ = [
    "ArmJointMap",
    "clamp_to_limits",
    "clip_positions_to_limits",
    "detect_arm_joint_qaddr",
    "detect_arm_joints",
    "load_mujoco_model",
    "load_mujoco_robot_model",
]
