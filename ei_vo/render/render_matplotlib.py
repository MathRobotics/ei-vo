"""Matplotlib-based 3D playback for MuJoCo models."""

from __future__ import annotations

import pathlib
import sys

import numpy as np

from ..config import (
    CameraSettings,
    PlaybackConfig,
    RecordingConfig,
    coerce_camera_settings,
    coerce_playback_config,
    normalize_runtime_config,
    resolve_recording_fps,
)
from ..core import FrameSequenceWriter, Trajectory
from ..modeling import (
    clip_positions_to_limits as _clip_positions_to_limits,
    detect_arm_joints,
    load_mujoco_model as _load_mujoco_model,
)

_IMAGE_SUFFIXES = {".png", ".pdf", ".svg", ".jpg", ".jpeg"}
_VIDEO_SUFFIXES = {".mp4", ".gif", ".webm", ".mov"}


def _resolve_figsize(
    *,
    dpi: int,
    record_size: tuple[int, int] | None,
) -> tuple[float, float]:
    if record_size is not None:
        width_px, height_px = record_size
        if width_px <= 0 or height_px <= 0:
            raise ValueError(f"record_size must contain positive integers. Got {record_size}.")
        return width_px / dpi, height_px / dpi
    return (7.5, 7.0)


def _geom_rgba(model, geom_id: int) -> np.ndarray:
    if hasattr(model, "geom_rgba"):
        return np.asarray(model.geom_rgba[geom_id], dtype=float)
    return np.array([0.7, 0.7, 0.7, 1.0], dtype=float)


def _geom_color(model, geom_id: int) -> tuple[float, float, float]:
    rgba = _geom_rgba(model, geom_id)
    return tuple(np.clip(rgba[:3], 0.0, 1.0))


def _supported_geom_ids(mj, model) -> tuple[int, ...]:
    supported_geom_types = {
        getattr(mj.mjtGeom, "mjGEOM_SPHERE", -1),
        getattr(mj.mjtGeom, "mjGEOM_BOX", -1),
        getattr(mj.mjtGeom, "mjGEOM_CYLINDER", -1),
        getattr(mj.mjtGeom, "mjGEOM_CAPSULE", -1),
        getattr(mj.mjtGeom, "mjGEOM_ELLIPSOID", -1),
    }
    return tuple(
        geom_id
        for geom_id in range(getattr(model, "ngeom", 0))
        if int(model.geom_type[geom_id]) in supported_geom_types
    )


def _snapshot_frame(mj, model, data, geom_ids: tuple[int, ...]) -> dict[str, object]:
    lines: list[tuple[np.ndarray, tuple[float, float, float], float]] = []
    scatter_points: list[np.ndarray] = []
    scatter_colors: list[tuple[float, float, float]] = []
    scatter_sizes: list[float] = []
    bounds: list[np.ndarray] = []

    cylinder_like = {
        getattr(mj.mjtGeom, "mjGEOM_CYLINDER", -1),
        getattr(mj.mjtGeom, "mjGEOM_CAPSULE", -1),
    }

    for geom_id in geom_ids:
        geom_type = int(model.geom_type[geom_id])
        geom_size = np.asarray(model.geom_size[geom_id], dtype=float)
        center = np.asarray(data.geom_xpos[geom_id], dtype=float)
        rotation = np.asarray(data.geom_xmat[geom_id], dtype=float).reshape(3, 3)
        color = _geom_color(model, geom_id)

        if geom_type in cylinder_like and geom_size.shape[0] >= 2 and geom_size[1] > 0.0:
            axis_direction = rotation[:, 2]
            half_length = float(geom_size[1])
            segment = np.vstack((center - axis_direction * half_length, center + axis_direction * half_length))
            lines.append((segment, color, 2.8))
            bounds.append(segment)
            continue

        scatter_points.append(center)
        scatter_colors.append(color)
        scatter_sizes.append(800.0 * max(float(np.max(geom_size)), 0.03))
        bounds.append(center[None, :])

    if bounds:
        bounds_array = np.concatenate(bounds, axis=0)
    else:
        bounds_array = np.zeros((1, 3), dtype=float)

    if scatter_points:
        points_array = np.vstack(scatter_points)
    else:
        points_array = np.zeros((0, 3), dtype=float)

    return {
        "lines": tuple(lines),
        "points": points_array,
        "colors": tuple(scatter_colors),
        "sizes": tuple(scatter_sizes),
        "bounds": bounds_array,
    }


def _compute_axis_limits(bounds: np.ndarray) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    mins = np.min(bounds, axis=0)
    maxs = np.max(bounds, axis=0)
    center = 0.5 * (mins + maxs)
    radius = max(float(np.max(maxs - mins)) * 0.55, 0.25)
    return (
        (center[0] - radius, center[0] + radius),
        (center[1] - radius, center[1] + radius),
        (center[2] - radius, center[2] + radius),
    )


def _camera_value(camera: object, key: str) -> float | None:
    settings = coerce_camera_settings(camera)
    if settings is None:
        return None
    value = getattr(settings, key, None)
    return None if value is None else float(value)


def _configure_axis(axis, *, title: str, limits, camera: object) -> None:
    (xlim, ylim, zlim) = limits
    axis.set_xlim(*xlim)
    axis.set_ylim(*ylim)
    axis.set_zlim(*zlim)
    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")
    axis.set_zlabel("z [m]")
    axis.set_title(title)
    axis.grid(True, alpha=0.3)
    if hasattr(axis, "set_box_aspect"):
        axis.set_box_aspect((1.0, 1.0, 1.0))

    elevation = _camera_value(camera, "elevation")
    azimuth = _camera_value(camera, "azimuth")
    if hasattr(axis, "view_init") and (elevation is not None or azimuth is not None):
        axis.view_init(elev=elevation if elevation is not None else 25.0, azim=azimuth if azimuth is not None else 45.0)


def _draw_frame(axis, *, frame: dict[str, object], title: str, limits, camera: object) -> None:
    axis.clear()

    for segment, color, linewidth in frame["lines"]:
        axis.plot(
            segment[:, 0],
            segment[:, 1],
            segment[:, 2],
            color=color,
            linewidth=linewidth,
        )

    points = frame["points"]
    if len(points) > 0:
        axis.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            c=list(frame["colors"]),
            s=list(frame["sizes"]),
            depthshade=False,
        )

    _configure_axis(axis, title=title, limits=limits, camera=camera)


def _import_pyplot(*, force_agg: bool):
    try:
        import matplotlib
        if force_agg and hasattr(matplotlib, "use") and "matplotlib.pyplot" not in sys.modules:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "The 'matplotlib' renderer requires the optional dependency 'matplotlib'."
        ) from exc
    return plt


def _resolve_record_output_path(record_path: str | pathlib.Path) -> tuple[pathlib.Path, str]:
    output_path = pathlib.Path(record_path)
    suffix = output_path.suffix.lower()
    if suffix in _VIDEO_SUFFIXES:
        return output_path, "video"
    if suffix in _IMAGE_SUFFIXES:
        return output_path, "image"
    return output_path.with_suffix(".png"), "image"


def _capture_figure_frame(figure) -> np.ndarray:
    canvas = getattr(figure, "canvas", None)
    if canvas is None or not hasattr(canvas, "draw") or not hasattr(canvas, "buffer_rgba"):
        raise RuntimeError("The active Matplotlib backend does not support offscreen frame capture.")

    canvas.draw()
    width, height = canvas.get_width_height()
    frame = np.asarray(canvas.buffer_rgba(), dtype=np.uint8)
    if frame.shape != (height, width, 4):
        frame = frame.reshape((height, width, 4))
    return frame[:, :, :3].copy()


def play_trajectory(
    model_path: str | pathlib.Path | None,
    trajectory: Trajectory | np.ndarray | list[list[float]] | list[float],
    *,
    playback: PlaybackConfig | None = None,
    camera: CameraSettings | None = None,
    recording: RecordingConfig | None = None,
    show: bool = True,
    title: str | None = None,
) -> None:
    """Render a MuJoCo trajectory with Matplotlib in 3D."""

    if model_path is None:
        raise ValueError("--model is required when using the matplotlib renderer.")

    playback = coerce_playback_config(playback)
    if playback.loop:
        raise ValueError("The 'matplotlib' renderer does not support loop playback.")

    pre_record_mode = None
    if recording is not None:
        _, pre_record_mode = _resolve_record_output_path(recording.path)

    plt = _import_pyplot(force_agg=pre_record_mode == "video")

    try:
        import mujoco as mj
    except ImportError as exc:
        raise RuntimeError(
            "The 'matplotlib' renderer requires the 'mujoco' package. Run `uv sync` in this project."
        ) from exc

    trajectory = Trajectory.coerce(trajectory)
    path = pathlib.Path(model_path)
    model = _load_mujoco_model(path)
    data = mj.MjData(model)
    arm_joints = detect_arm_joints(model, expected_dof=trajectory.dof)
    positions = _clip_positions_to_limits(trajectory.q, arm_joints.limits)
    geom_ids = _supported_geom_ids(mj, model)
    if not geom_ids:
        raise RuntimeError("The 'matplotlib' renderer found no supported MuJoCo geoms to draw.")

    frames: list[dict[str, object]] = []
    bounds_per_frame: list[np.ndarray] = []
    for row in positions:
        for qpos_address, value in zip(arm_joints.qpos_addresses, row):
            data.qpos[qpos_address] = float(value)
        mj.mj_forward(model, data)
        frame = _snapshot_frame(mj, model, data, geom_ids)
        frames.append(frame)
        bounds_per_frame.append(frame["bounds"])

    limits = _compute_axis_limits(np.concatenate(bounds_per_frame, axis=0))

    dpi = 100
    figsize = _resolve_figsize(dpi=dpi, record_size=None if recording is None else recording.size)
    figure = plt.figure(figsize=figsize, constrained_layout=True)
    axis = figure.add_subplot(111, projection="3d")
    base_title = title or "Matplotlib 3D Playback"
    output_path = None
    record_mode = None
    if recording is not None:
        output_path, record_mode = _resolve_record_output_path(recording.path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if recording.frames_dir is not None and record_mode != "video":
            raise ValueError("record_frames_dir is only supported for video recording outputs.")

    effective_show = bool(show and record_mode != "video")
    animate = effective_show or record_mode == "video"
    frame_indices = range(trajectory.steps) if animate else (trajectory.steps - 1,)
    writer = None
    try:
        if record_mode == "video" and output_path is not None and recording is not None:
            writer = FrameSequenceWriter(
                output_path,
                fps=resolve_recording_fps(playback, recording),
                frames_dir=recording.frames_dir,
                temp_prefix="ei_vo_matplotlib_",
            )

        for frame_index in frame_indices:
            _draw_frame(
                axis,
                frame=frames[frame_index],
                title=f"{base_title} ({frame_index + 1}/{trajectory.steps})",
                limits=limits,
                camera=camera,
            )
            if writer is not None:
                writer.append_data(_capture_figure_frame(figure))
            if effective_show and frame_index < trajectory.steps - 1:
                plt.pause(max(1e-4, playback.step_dt))

        if record_mode == "image" and output_path is not None:
            figure.savefig(output_path.as_posix(), dpi=dpi)

        if effective_show:
            plt.show()
    finally:
        if writer is not None:
            writer.close()
        plt.close(figure)


def play(
    model_path: str | pathlib.Path | None,
    traj,
    slow: float = 1.0,
    hz: float = 240.0,
    camera=None,
    loop: bool = False,
    record_path: str | pathlib.Path | None = None,
    record_fps: float | None = None,
    record_size: tuple[int, int] | None = None,
    record_frames_dir: str | pathlib.Path | None = None,
    *,
    show: bool = True,
    title: str | None = None,
    kinematics_backend: str | None = None,
    kinematics_model_path: str | pathlib.Path | None = None,
    base_link: str | None = None,
    end_link: str | None = None,
):
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
        show=show,
        title=title,
    )


__all__ = [
    "play",
    "play_trajectory",
]
