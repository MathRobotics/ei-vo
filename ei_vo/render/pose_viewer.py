"""Interactive MuJoCo model visualisation utilities."""
from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Iterable, Optional, Sequence

import mujoco as mj
import mujoco.viewer as viewer
import numpy as np

from .render_mj import clamp_to_limits, detect_arm_joint_qaddr


@dataclass
class JointInfo:
    """Metadata describing a controllable joint."""

    qaddr: int
    name: str
    limit_min: Optional[float]
    limit_max: Optional[float]


def _gather_joint_info(model: mj.MjModel, expected_dof: Optional[int]) -> list[JointInfo]:
    """Detect controllable arm joints and collect metadata."""

    qaddrs = detect_arm_joint_qaddr(model, expected_dof)
    infos: list[JointInfo] = []
    for qaddr in qaddrs:
        joint_indices = np.where(model.jnt_qposadr == qaddr)[0]
        if len(joint_indices) == 0:
            name = ""
            limit_min = None
            limit_max = None
        else:
            j_id = int(joint_indices[0])
            raw_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, j_id)
            name = raw_name or ""
            low, high = model.jnt_range[j_id]
            if low < high:
                limit_min = float(low)
                limit_max = float(high)
            else:
                limit_min = None
                limit_max = None
        infos.append(JointInfo(qaddr=qaddr, name=name, limit_min=limit_min, limit_max=limit_max))
    return infos


def _format_joint_name(index: int, info: JointInfo) -> str:
    if info.name:
        return f"{index + 1}: {info.name}"
    return f"Joint {index + 1}"


def _apply_pose(model: mj.MjModel, data: mj.MjData, infos: Sequence[JointInfo], values: np.ndarray) -> None:
    """Write pose values into ``data`` and forward the model."""

    for info, value in zip(infos, values):
        data.qpos[info.qaddr] = float(value)
    mj.mj_forward(model, data)


def _overlay_for_pose(handle: viewer.Handle, infos: Sequence[JointInfo], pose: np.ndarray,
                      selected: int, step_size: float) -> None:
    """Update the overlay text to reflect the current pose."""

    lines = []
    for idx, (info, value) in enumerate(zip(infos, pose)):
        marker = "➤" if idx == selected else "  "
        angle_deg = math.degrees(value)
        if info.limit_min is not None and info.limit_max is not None:
            min_deg = math.degrees(info.limit_min)
            max_deg = math.degrees(info.limit_max)
            limit_desc = f"[{min_deg:.1f}, {max_deg:.1f}]°"
        else:
            limit_desc = "(free)"
        lines.append(f"{marker} {_format_joint_name(idx, info):<18} {angle_deg:+7.2f}° {limit_desc}")

    body = "\n".join(lines)
    handle.set_texts([
        (
            mj.mjtFontScale.mjFONTSCALE_150,
            mj.mjtGridPos.mjGRID_TOPLEFT,
            "Interactive pose editor",
            f"Step: {math.degrees(step_size):.2f}°  (UP/DOWN adjust, LEFT/RIGHT select)",
        ),
        (
            mj.mjtFontScale.mjFONTSCALE_100,
            mj.mjtGridPos.mjGRID_TOPLEFT,
            "[ / ] step ×0.5/×2 | R reset | 0 zero pose | ESC close",
            "",
        ),
        (
            mj.mjtFontScale.mjFONTSCALE_100,
            mj.mjtGridPos.mjGRID_BOTTOMLEFT,
            body,
            "",
        ),
    ])


def _clamp_value(value: float, info: JointInfo) -> float:
    if info.limit_min is None or info.limit_max is None:
        return value
    return float(np.clip(value, info.limit_min, info.limit_max))


def inspect_pose(
    model_path: str,
    *,
    expected_dof: Optional[int] = None,
    initial_pose: Optional[Iterable[float]] = None,
    step_deg: float = 5.0,
    clamp: bool = True,
) -> None:
    """Launch an interactive viewer for tweaking joint angles.

    Parameters
    ----------
    model_path:
        Path to the MJCF model to open.
    expected_dof:
        Optional number of arm joints to control. When provided the viewer will
        consider only the first ``expected_dof`` arm hinge joints detected in the
        model.
    initial_pose:
        Optional iterable providing the initial joint configuration in radians.
        When omitted the pose defaults to zero (subject to joint limits).
    step_deg:
        Increment/decrement step in degrees for keyboard interaction.
    clamp:
        When ``True`` (default) joint movements respect the limits provided in
        the MJCF model.
    """

    if step_deg <= 0.0:
        raise ValueError("step_deg must be positive")

    model = mj.MjModel.from_xml_path(model_path)
    data = mj.MjData(model)

    infos = _gather_joint_info(model, expected_dof)
    dof = len(infos)

    if dof == 0:
        raise RuntimeError("No controllable arm joints detected in the model.")

    pose = np.zeros(dof, dtype=float)

    if initial_pose is not None:
        arr = np.asarray(list(initial_pose), dtype=float)
        if arr.shape != (dof,):
            raise ValueError(
                f"initial_pose must contain {dof} elements (received shape {arr.shape})"
            )
        pose[:] = arr

    if clamp:
        pose = clamp_to_limits(model, [info.qaddr for info in infos], pose[None, :])[0]

    _apply_pose(model, data, infos, pose)

    selected = 0
    step_size = math.radians(step_deg)
    min_step = math.radians(0.1)
    max_step = math.radians(90.0)
    default_pose = pose.copy()

    def update_overlay(handle: viewer.Handle) -> None:
        _overlay_for_pose(handle, infos, pose, selected, step_size)

    def apply_and_refresh(handle: viewer.Handle) -> None:
        _apply_pose(model, data, infos, pose)
        update_overlay(handle)

    def on_key(key: int) -> None:
        nonlocal selected, step_size, pose
        handle = viewer_handle
        if handle is None:
            return

        if key in (viewer.glfw.KEY_RIGHT, viewer.glfw.KEY_D):
            selected = (selected + 1) % dof
            update_overlay(handle)
        elif key in (viewer.glfw.KEY_LEFT, viewer.glfw.KEY_A):
            selected = (selected - 1) % dof
            update_overlay(handle)
        elif key in (viewer.glfw.KEY_UP, viewer.glfw.KEY_W, viewer.glfw.KEY_KP_ADD):
            pose[selected] += step_size
            if clamp:
                pose[selected] = _clamp_value(pose[selected], infos[selected])
            apply_and_refresh(handle)
        elif key in (viewer.glfw.KEY_DOWN, viewer.glfw.KEY_S, viewer.glfw.KEY_KP_SUBTRACT):
            pose[selected] -= step_size
            if clamp:
                pose[selected] = _clamp_value(pose[selected], infos[selected])
            apply_and_refresh(handle)
        elif key == viewer.glfw.KEY_LEFT_BRACKET:
            step_size = float(np.clip(step_size * 0.5, min_step, max_step))
            update_overlay(handle)
        elif key == viewer.glfw.KEY_RIGHT_BRACKET:
            step_size = float(np.clip(step_size * 2.0, min_step, max_step))
            update_overlay(handle)
        elif key in (viewer.glfw.KEY_R, viewer.glfw.KEY_BACKSPACE):
            pose[:] = default_pose
            if clamp:
                pose[selected] = _clamp_value(pose[selected], infos[selected])
            apply_and_refresh(handle)
        elif key == viewer.glfw.KEY_0:
            pose[:] = 0.0
            if clamp:
                for i, info in enumerate(infos):
                    pose[i] = _clamp_value(pose[i], info)
            apply_and_refresh(handle)

    viewer_handle: Optional[viewer.Handle] = None

    viewer_handle = viewer.launch_passive(
        model,
        data,
        key_callback=on_key,
        show_left_ui=False,
        show_right_ui=True,
    )

    with viewer_handle as handle:
        update_overlay(handle)
        while handle.is_running():
            handle.sync(state_only=True)
            time.sleep(0.01)
