"""MuJoCo playback utilities."""

from __future__ import annotations

import contextlib
import pathlib
import time
from typing import Mapping

import mujoco as mj
import mujoco.viewer as viewer
import numpy as np

from ..config import (
    CameraSettings,
    PlaybackConfig,
    RecordingConfig,
    coerce_playback_config,
    coerce_camera_settings,
    normalize_runtime_config,
)
from ..core import FrameSequenceWriter, Trajectory
from ..modeling import (
    ArmJointMap,
    clamp_to_limits,
    clip_positions_to_limits as _clip_positions_to_limits,
    detect_arm_joint_qaddr,
    detect_arm_joints,
    load_mujoco_model as _load_mujoco_model,
    load_mujoco_robot_model as load_robot_model,
)


def _apply_camera_settings(
    camera: object,
    settings: CameraSettings | Mapping[str, object] | None,
) -> None:
    settings = coerce_camera_settings(settings)
    if settings is None:
        return

    if settings.distance is not None:
        camera.distance = float(settings.distance)
    if settings.azimuth is not None:
        camera.azimuth = float(settings.azimuth)
    if settings.elevation is not None:
        camera.elevation = float(settings.elevation)
    if settings.lookat is not None:
        camera.lookat[:] = settings.lookat


def _configure_default_camera(model: mj.MjModel, active_viewer: object) -> None:
    if hasattr(mj, "mjv_defaultFreeCamera"):
        mj.mjv_defaultFreeCamera(model, active_viewer.cam)
        return

    mj.mjv_defaultCamera(active_viewer.cam)
    if hasattr(model, "stat"):
        center = getattr(model.stat, "center", None)
        if center is not None:
            active_viewer.cam.lookat[:] = center
        extent = getattr(model.stat, "extent", None)
        if extent:
            active_viewer.cam.distance = extent


def _init_recording(
    model: mj.MjModel,
    dt: float,
    record_path: str | pathlib.Path,
    record_fps: float | None,
    record_size: tuple[int, int] | None,
    record_frames_dir: str | pathlib.Path | None,
):
    """Initialise offscreen rendering resources."""

    recording = RecordingConfig(path=record_path, fps=record_fps, size=record_size, frames_dir=record_frames_dir)
    path = recording.path
    if path.suffix == "":
        path = path.with_suffix(".mp4")
    path.parent.mkdir(parents=True, exist_ok=True)

    width, height = recording.size if recording.size is not None else (1280, 720)

    vis = getattr(model, "vis", None)
    global_vis = getattr(vis, "global_", None) if vis is not None else None
    if global_vis is not None and hasattr(global_vis, "offwidth") and hasattr(global_vis, "offheight"):
        global_vis.offwidth = max(int(global_vis.offwidth), int(width))
        global_vis.offheight = max(int(global_vis.offheight), int(height))

    renderer = mj.Renderer(model, height=height, width=width)
    camera = mj.MjvCamera()
    mj.mjv_defaultCamera(camera)
    fps = recording.fps if recording.fps is not None else (1.0 / max(dt, 1e-9))
    writer = FrameSequenceWriter(path, fps=fps, frames_dir=recording.frames_dir)
    return renderer, camera, writer


def play_trajectory(
    model_path: str | pathlib.Path,
    trajectory: Trajectory | np.ndarray | list[list[float]] | list[float],
    *,
    playback: PlaybackConfig | None = None,
    camera: CameraSettings | Mapping[str, object] | None = None,
    recording: RecordingConfig | None = None,
) -> None:
    """Play a trajectory on top of a MuJoCo model."""

    path = pathlib.Path(model_path)
    model = _load_mujoco_model(path)
    data = mj.MjData(model)
    traj = Trajectory.coerce(trajectory)
    arm_joints = detect_arm_joints(model, expected_dof=traj.dof)
    playback = coerce_playback_config(playback)
    positions = _clip_positions_to_limits(traj.q, arm_joints.limits)

    record_renderer = None
    record_camera = None
    record_writer = None
    if recording is not None:
        record_renderer, record_camera, record_writer = _init_recording(
            model,
            playback.step_dt,
            recording.path,
            recording.fps,
            recording.size,
            recording.frames_dir,
        )

    with contextlib.ExitStack() as stack:
        active_viewer = stack.enter_context(viewer.launch_passive(model, data))
        if record_writer is not None:
            stack.callback(record_writer.close)
        if record_renderer is not None:
            stack.callback(record_renderer.close)

        if camera is None:
            _configure_default_camera(model, active_viewer)
        else:
            _apply_camera_settings(active_viewer.cam, camera)

        if record_camera is not None:
            _apply_camera_settings(record_camera, CameraSettings.from_camera(active_viewer.cam))

        while active_viewer.is_running():
            for row in positions:
                for qpos_address, value in zip(arm_joints.qpos_addresses, row):
                    data.qpos[qpos_address] = float(value)
                mj.mj_forward(model, data)

                if record_renderer is not None and record_camera is not None and record_writer is not None:
                    _apply_camera_settings(record_camera, CameraSettings.from_camera(active_viewer.cam))
                    record_renderer.update_scene(data, camera=record_camera)
                    frame = record_renderer.render()
                    if frame.dtype != np.uint8:
                        frame = np.clip(frame * 255.0, 0.0, 255.0).astype(np.uint8)
                    record_writer.append_data(frame)

                active_viewer.sync()
                time.sleep(playback.step_dt)

            if not playback.loop:
                break

        while active_viewer.is_running():
            active_viewer.sync()
            time.sleep(0.01)


def play(
    model_path: str | pathlib.Path,
    traj: Trajectory | np.ndarray | list[list[float]] | list[float],
    slow: float = 1.0,
    hz: float = 240.0,
    camera: CameraSettings | Mapping[str, object] | None = None,
    loop: bool = False,
    record_path: str | pathlib.Path | None = None,
    record_fps: float | None = None,
    record_size: tuple[int, int] | None = None,
    record_frames_dir: str | pathlib.Path | None = None,
    *,
    kinematics_backend: str | None = None,
    kinematics_model_path: str | pathlib.Path | None = None,
    base_link: str | None = None,
    end_link: str | None = None,
) -> None:
    """Compatibility wrapper around :func:`play_trajectory`."""

    del kinematics_backend, kinematics_model_path, base_link, end_link

    playback_config, camera_settings, recording = normalize_runtime_config(
        hz=hz,
        slow=slow,
        loop=loop,
        camera=camera,
        record_path=record_path,
        record_fps=record_fps,
        record_size=record_size,
        record_frames_dir=record_frames_dir,
    )
    play_trajectory(
        model_path,
        traj,
        playback=playback_config,
        camera=camera_settings,
        recording=recording,
    )


__all__ = [
    "ArmJointMap",
    "CameraSettings",
    "PlaybackConfig",
    "RecordingConfig",
    "_init_recording",
    "clamp_to_limits",
    "detect_arm_joint_qaddr",
    "detect_arm_joints",
    "load_robot_model",
    "play",
    "play_trajectory",
]
