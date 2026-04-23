"""Matplotlib-based 3D playback for URDF models."""

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
from ..modeling import compute_link_poses, load_urdf_scene

_IMAGE_SUFFIXES = {".png", ".pdf", ".svg", ".jpg", ".jpeg"}
_VIDEO_SUFFIXES = {".mp4", ".gif", ".webm", ".mov"}
_KNOWN_NON_INTERACTIVE_BACKENDS = frozenset({"agg", "cairo", "pdf", "pgf", "ps", "svg", "template"})
_SKELETON_COLOR = (0.35, 0.35, 0.35)


def _require_urdf_model_path(model_path: str | pathlib.Path | None) -> pathlib.Path:
    if model_path is None:
        raise ValueError("--model is required when using the matplotlib renderer.")
    path = pathlib.Path(model_path)
    if path.suffix.lower() != ".urdf":
        raise ValueError(f"The 'matplotlib' renderer only supports URDF models. Got {path!s}.")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


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


def _normalize_backend_name(backend: object) -> str:
    backend_name = "" if backend is None else str(backend).strip().lower()
    if backend_name.startswith("module://"):
        backend_name = backend_name.removeprefix("module://")
    return backend_name


def _is_non_interactive_backend_name(backend_name: str) -> bool:
    if not backend_name:
        return False
    if backend_name in _KNOWN_NON_INTERACTIVE_BACKENDS:
        return True
    return backend_name.rsplit(".", 1)[-1] in _KNOWN_NON_INTERACTIVE_BACKENDS


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


def _import_pyplot(*, force_agg: bool):
    try:
        import matplotlib

        if force_agg and hasattr(matplotlib, "use") and "matplotlib.pyplot" not in sys.modules:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("The 'matplotlib' renderer requires the optional dependency 'matplotlib'.") from exc
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


def _supports_live_show(figure) -> bool:
    canvas = getattr(figure, "canvas", None)
    manager = getattr(canvas, "manager", None)
    if manager is not None:
        try:
            from matplotlib.backend_bases import FigureManagerBase
        except Exception:
            pass
        else:
            return type(manager).show is not FigureManagerBase.show

    matplotlib = sys.modules.get("matplotlib")
    get_backend = getattr(matplotlib, "get_backend", None)
    if callable(get_backend):
        return not _is_non_interactive_backend_name(_normalize_backend_name(get_backend()))
    return True


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


def _snapshot_frame(scene, row: np.ndarray) -> dict[str, object]:
    link_poses = compute_link_poses(scene, row)
    lines: list[tuple[np.ndarray, tuple[float, float, float], float]] = []
    scatter_points: list[np.ndarray] = []
    scatter_colors: list[tuple[float, float, float]] = []
    scatter_sizes: list[float] = []
    bounds: list[np.ndarray] = []

    for parent_link, joints in scene.child_joints.items():
        parent_pose = link_poses.get(parent_link)
        if parent_pose is None:
            continue
        parent_position = np.asarray(parent_pose[:3, 3], dtype=float)
        for joint in joints:
            child_pose = link_poses.get(joint.child_link)
            if child_pose is None:
                continue
            child_position = np.asarray(child_pose[:3, 3], dtype=float)
            segment = np.vstack((parent_position, child_position))
            lines.append((segment, _SKELETON_COLOR, 1.5))
            bounds.append(segment)

    for link_name, visuals in scene.link_visuals.items():
        link_pose = link_poses.get(link_name)
        if link_pose is None:
            continue
        for visual in visuals:
            pose = link_pose @ visual.origin
            center = np.asarray(pose[:3, 3], dtype=float)
            color = tuple(np.clip(np.asarray(visual.rgba[:3], dtype=float), 0.0, 1.0))

            if visual.geometry_type == "cylinder" and visual.length is not None and visual.length > 0.0:
                axis = np.asarray(pose[:3, :3], dtype=float)[:, 2]
                half_length = 0.5 * float(visual.length)
                segment = np.vstack((center - axis * half_length, center + axis * half_length))
                lines.append((segment, color, 3.0))
                bounds.append(segment)
                continue

            scatter_points.append(center)
            scatter_colors.append(color)
            if visual.geometry_type == "box" and visual.size is not None:
                scale = max(float(np.max(np.asarray(visual.size, dtype=float))), 0.03)
            else:
                scale = max(float(visual.radius or 0.03), 0.03)
            scatter_sizes.append(1200.0 * scale)
            bounds.append(center[None, :])

    if not bounds:
        all_positions = np.vstack([np.asarray(pose[:3, 3], dtype=float) for pose in link_poses.values()])
        bounds_array = all_positions if len(all_positions) > 0 else np.zeros((1, 3), dtype=float)
    else:
        bounds_array = np.concatenate(bounds, axis=0)

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
    """Render a URDF trajectory with Matplotlib in 3D."""

    path = _require_urdf_model_path(model_path)
    playback = coerce_playback_config(playback)
    if playback.loop:
        raise ValueError("The 'matplotlib' renderer does not support loop playback.")

    pre_record_mode = None
    if recording is not None:
        _, pre_record_mode = _resolve_record_output_path(recording.path)

    plt = _import_pyplot(force_agg=pre_record_mode == "video")
    trajectory = Trajectory.coerce(trajectory)
    scene = load_urdf_scene(path, expected_dof=trajectory.dof)
    positions = scene.clamp(trajectory.q)

    frames = [_snapshot_frame(scene, row) for row in positions]
    limits = _compute_axis_limits(np.concatenate([frame["bounds"] for frame in frames], axis=0))

    dpi = 100
    figsize = _resolve_figsize(dpi=dpi, record_size=None if recording is None else recording.size)
    figure = plt.figure(figsize=figsize, constrained_layout=True)
    axis = figure.add_subplot(111, projection="3d")
    base_title = title or "Matplotlib URDF Playback"
    output_path = None
    record_mode = None
    if recording is not None:
        output_path, record_mode = _resolve_record_output_path(recording.path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if recording.frames_dir is not None and record_mode != "video":
            raise ValueError("record_frames_dir is only supported for video recording outputs.")

    effective_show = bool(show and record_mode != "video" and _supports_live_show(figure))
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


__all__ = ["play", "play_trajectory"]
