"""MeshCat-based playback backend for URDF models."""

from __future__ import annotations

import pathlib
import time
import webbrowser
from collections.abc import Mapping

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
from ..core import Trajectory


def _import_meshcat():
    try:
        import meshcat
    except ImportError as exc:
        raise RuntimeError("The 'meshcat' renderer requires the optional dependency 'meshcat'.") from exc
    return meshcat


def _import_pinocchio_visualizer():
    try:
        import pinocchio as pin
        from pinocchio.visualize import MeshcatVisualizer
    except ImportError as exc:
        raise RuntimeError(
            "The 'meshcat' renderer requires the optional dependency 'pin' for URDF visual playback."
        ) from exc
    return pin, MeshcatVisualizer


def _create_visualizer(meshcat_module):
    return meshcat_module.Visualizer()


def _require_urdf_model_path(model_path: str | pathlib.Path | None) -> pathlib.Path:
    if model_path is None:
        raise ValueError("--model is required when using the meshcat renderer.")
    path = pathlib.Path(model_path)
    if path.suffix.lower() != ".urdf":
        raise ValueError(f"The 'meshcat' renderer only supports URDF models. Got {path!s}.")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path.resolve()


def _pinocchio_package_dirs(model_path: pathlib.Path) -> list[str]:
    model_dir = model_path.expanduser().resolve(strict=False).parent
    package_dirs: list[str] = []
    seen: set[str] = set()
    for candidate in (model_dir, *model_dir.parents):
        candidate_str = candidate.as_posix()
        if candidate_str in seen:
            continue
        seen.add(candidate_str)
        package_dirs.append(candidate_str)
    return package_dirs


def _clip_positions_to_pinocchio_limits(
    q: np.ndarray,
    lower_limits: np.ndarray,
    upper_limits: np.ndarray,
) -> np.ndarray:
    positions = np.asarray(q, dtype=float).copy()
    lower = np.asarray(lower_limits, dtype=float).reshape(-1)
    upper = np.asarray(upper_limits, dtype=float).reshape(-1)
    if positions.ndim != 2:
        raise ValueError(f"Trajectory positions must be 2D. Got {positions.shape}.")
    if positions.shape[1] != lower.shape[0] or positions.shape[1] != upper.shape[0]:
        raise ValueError(
            "Trajectory dof "
            f"({positions.shape[1]}) does not match Pinocchio model dof ({lower.shape[0]})."
        )
    for index, (lower_bound, upper_bound) in enumerate(zip(lower, upper)):
        if np.isfinite(lower_bound) and np.isfinite(upper_bound) and lower_bound < upper_bound:
            positions[:, index] = np.clip(positions[:, index], lower_bound, upper_bound)
    return positions


def _make_transform(
    translation: np.ndarray,
    rotation: np.ndarray | None = None,
    scale: np.ndarray | None = None,
) -> np.ndarray:
    transform = np.eye(4, dtype=float)
    if rotation is not None:
        transform[:3, :3] = np.asarray(rotation, dtype=float).reshape(3, 3)
    if scale is not None:
        transform[:3, :3] = transform[:3, :3] @ np.diag(np.asarray(scale, dtype=float))
    transform[:3, 3] = np.asarray(translation, dtype=float)
    return transform


def _rotation_x(theta: float) -> np.ndarray:
    cos_theta = float(np.cos(theta))
    sin_theta = float(np.sin(theta))
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cos_theta, -sin_theta],
            [0.0, sin_theta, cos_theta],
        ],
        dtype=float,
    )


_MESHCAT_CAMERA_ROOT_PATH = "/Cameras/default"
_MESHCAT_CAMERA_OBJECT_PATH = "/Cameras/default/rotated/<object>"
_MESHCAT_DEFAULT_CAMERA_POSITION_ROTATED = np.array([3.0, 1.0, 0.0], dtype=float)
_MESHCAT_ROTATED_TO_WORLD = _rotation_x(np.pi / 2.0)
_MESHCAT_WORLD_TO_ROTATED = _rotation_x(-np.pi / 2.0)


def _camera_settings_from_world_offset(world_offset: np.ndarray) -> CameraSettings:
    offset = np.asarray(world_offset, dtype=float).reshape(3)
    distance = float(np.linalg.norm(offset))
    if distance <= 1e-12:
        return CameraSettings(distance=0.0, azimuth=0.0, elevation=0.0, lookat=np.zeros(3))
    azimuth = float(np.degrees(np.arctan2(-offset[1], -offset[0])))
    elevation = float(np.degrees(np.arctan2(-offset[2], np.hypot(offset[0], offset[1]))))
    return CameraSettings(distance=distance, azimuth=azimuth, elevation=elevation, lookat=np.zeros(3))


_MESHCAT_DEFAULT_CAMERA_SETTINGS = _camera_settings_from_world_offset(
    _MESHCAT_ROTATED_TO_WORLD @ _MESHCAT_DEFAULT_CAMERA_POSITION_ROTATED
)


def _resolve_meshcat_camera_settings(
    camera: object | Mapping[str, object] | None,
) -> CameraSettings | None:
    settings = coerce_camera_settings(camera)
    if settings is None:
        return None
    defaults = _MESHCAT_DEFAULT_CAMERA_SETTINGS
    return CameraSettings(
        distance=defaults.distance if settings.distance is None else float(settings.distance),
        azimuth=defaults.azimuth if settings.azimuth is None else float(settings.azimuth),
        elevation=defaults.elevation if settings.elevation is None else float(settings.elevation),
        lookat=np.zeros(3, dtype=float) if settings.lookat is None else np.asarray(settings.lookat, dtype=float),
    )


def _camera_world_offset(settings: CameraSettings) -> np.ndarray:
    azimuth = float(np.deg2rad(settings.azimuth))
    elevation = float(np.deg2rad(settings.elevation))
    cos_elevation = float(np.cos(elevation))
    return np.array(
        [
            -float(settings.distance) * cos_elevation * float(np.cos(azimuth)),
            -float(settings.distance) * cos_elevation * float(np.sin(azimuth)),
            -float(settings.distance) * float(np.sin(elevation)),
        ],
        dtype=float,
    )


def _apply_camera_settings(
    visualizer,
    camera: object | Mapping[str, object] | None,
) -> None:
    settings = _resolve_meshcat_camera_settings(camera)
    if settings is None:
        return

    visualizer[_MESHCAT_CAMERA_ROOT_PATH].set_transform(
        _make_transform(np.asarray(settings.lookat, dtype=float))
    )
    rotated_offset = _MESHCAT_WORLD_TO_ROTATED @ _camera_world_offset(settings)
    rotated_offset[np.abs(rotated_offset) <= 1e-12] = 0.0
    visualizer[_MESHCAT_CAMERA_OBJECT_PATH].set_property(
        "position",
        [float(value) for value in rotated_offset],
    )


def _save_html(visualizer, record_path: str | pathlib.Path) -> pathlib.Path:
    path = pathlib.Path(record_path)
    if path.suffix == "":
        path = path.with_suffix(".html")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not hasattr(visualizer, "static_html"):
        raise RuntimeError("The installed meshcat.Visualizer does not support static_html().")
    path.write_text(visualizer.static_html(), encoding="utf-8")
    return path


def _resolve_record_html_path(record_path: str | pathlib.Path) -> pathlib.Path:
    path = pathlib.Path(record_path)
    return path if path.suffix.lower() == ".html" else path.with_suffix(".html")


def _export_recording_artifacts(
    visualizer,
    *,
    recording: RecordingConfig,
    recorder: "_AnimationRecorder | None",
) -> pathlib.Path:
    if recorder is not None:
        recorder.begin_frame(None)
        recorder.apply(visualizer)
    return _save_html(visualizer, _resolve_record_html_path(recording.path))


def _open_standalone_recording_html(html_path: pathlib.Path) -> None:
    try:
        webbrowser.open(html_path.resolve(strict=False).as_uri(), new=2)
    except Exception:
        return


class _AnimationRecorder:
    def __init__(self, meshcat_module, *, fps: float) -> None:
        animation_module = getattr(meshcat_module, "animation", None)
        animation_cls = getattr(animation_module, "Animation", None) if animation_module is not None else None
        self.animation = None if animation_cls is None else animation_cls(default_framerate=fps)
        self._frame_index: int | None = None

    @property
    def enabled(self) -> bool:
        return self.animation is not None

    def begin_frame(self, frame_index: int | None) -> None:
        self._frame_index = None if frame_index is None else int(frame_index)

    def capture_transform(self, node, transform: np.ndarray) -> None:
        if self.animation is None or self._frame_index is None:
            return
        with self.animation.at_frame(node, self._frame_index) as frame_visualizer:
            frame_visualizer.set_transform(np.asarray(transform, dtype=float))

    def apply(self, visualizer) -> bool:
        if self.animation is None:
            return False
        visualizer.set_animation(self.animation, play=True, repetitions=1)
        return True


class _RecordingNode:
    def __init__(self, node, recorder: _AnimationRecorder | None) -> None:
        self._node = node
        self._recorder = recorder

    @property
    def path(self):
        return self._node.path

    def __getitem__(self, key):
        return _RecordingNode(self._node[key], self._recorder)

    def set_transform(self, matrix=np.eye(4)):
        self._node.set_transform(matrix)
        if self._recorder is not None:
            self._recorder.capture_transform(self, matrix)

    def __getattr__(self, name):
        return getattr(self._node, name)


def _build_recording_visualizer(
    meshcat_module,
    playback: PlaybackConfig,
    *,
    recording: RecordingConfig,
):
    recorder = _AnimationRecorder(meshcat_module, fps=resolve_recording_fps(playback, recording))
    return recorder if recorder.enabled else None


def _play_with_pinocchio(
    path: pathlib.Path,
    trajectory: Trajectory,
    playback: PlaybackConfig,
    *,
    camera: CameraSettings | Mapping[str, object] | None,
    open_browser: bool,
    root_path: str,
    recording: RecordingConfig | None,
) -> None:
    meshcat = _import_meshcat()
    pin, MeshcatVisualizer = _import_pinocchio_visualizer()

    model, collision_model, visual_model = pin.buildModelsFromUrdf(
        path.as_posix(),
        package_dirs=_pinocchio_package_dirs(path),
    )
    model_dof = int(getattr(model, "nq"))
    if model_dof != trajectory.dof:
        raise ValueError(
            f"Trajectory dof ({trajectory.dof}) does not match Pinocchio model dof ({model_dof})."
        )

    lower_limits = getattr(model, "lowerPositionLimit", np.full(model_dof, -np.inf))
    upper_limits = getattr(model, "upperPositionLimit", np.full(model_dof, np.inf))
    positions = _clip_positions_to_pinocchio_limits(trajectory.q, lower_limits, upper_limits)

    visualizer = _create_visualizer(meshcat)
    recorder = _build_recording_visualizer(meshcat, playback, recording=recording) if recording else None
    recording_visualizer = _RecordingNode(visualizer, recorder) if recorder is not None else visualizer
    visualizer_wrapper = MeshcatVisualizer(model, collision_model, visual_model)
    visualizer_wrapper.initViewer(
        viewer=recording_visualizer,
        open=open_browser and recording is None,
        loadModel=False,
    )
    _apply_camera_settings(recording_visualizer, camera)
    visualizer_wrapper.loadViewerModel(rootNodeName=root_path or "pinocchio")

    exported = False
    while True:
        for frame_index, row in enumerate(positions):
            if recorder is not None:
                recorder.begin_frame(frame_index if not exported else None)
            visualizer_wrapper.display(np.asarray(row, dtype=float))
            time.sleep(playback.step_dt)

        if recording is not None and not exported:
            saved_html = _export_recording_artifacts(
                recording_visualizer,
                recording=recording,
                recorder=recorder,
            )
            if open_browser:
                _open_standalone_recording_html(saved_html)
            exported = True

        if not playback.loop:
            break


def play_trajectory(
    model_path: str | pathlib.Path | None,
    trajectory: Trajectory | np.ndarray | list[list[float]] | list[float],
    *,
    playback: PlaybackConfig | None = None,
    camera: CameraSettings | Mapping[str, object] | None = None,
    recording: RecordingConfig | None = None,
    open_browser: bool = True,
    root_path: str = "",
) -> None:
    """Render a URDF trajectory in MeshCat using normalized runtime config objects."""

    path = _require_urdf_model_path(model_path)
    playback = coerce_playback_config(playback)
    if recording is not None:
        if recording.size is not None:
            raise ValueError("record_size is not supported for MeshCat HTML export.")
        if recording.frames_dir is not None:
            raise ValueError("record_frames_dir is not supported for MeshCat HTML export.")

    _play_with_pinocchio(
        path,
        Trajectory.coerce(trajectory),
        playback,
        camera=camera,
        open_browser=open_browser,
        root_path=root_path,
        recording=recording,
    )


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
    open_browser: bool = True,
    root_path: str = "",
    kinematics_backend: str | None = None,
    kinematics_model_path: str | pathlib.Path | None = None,
    base_link: str | None = None,
    end_link: str | None = None,
) -> None:
    """Render a URDF trajectory in MeshCat."""

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
        open_browser=open_browser,
        root_path=root_path,
    )


__all__ = ["play", "play_trajectory"]
