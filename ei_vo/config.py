"""Backend-agnostic playback and recording configuration."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np


@dataclass(slots=True)
class CameraSettings:
    """Viewer camera configuration shared by render backends."""

    distance: float | None = None
    azimuth: float | None = None
    elevation: float | None = None
    lookat: np.ndarray | tuple[float, float, float] | None = None

    def __post_init__(self) -> None:
        if self.lookat is not None:
            lookat = np.asarray(self.lookat, dtype=float)
            if lookat.shape != (3,):
                raise ValueError(f"lookat must have shape (3,). Got {lookat.shape}.")
            self.lookat = lookat

    def to_dict(self) -> dict[str, object]:
        return {
            "distance": None if self.distance is None else float(self.distance),
            "azimuth": None if self.azimuth is None else float(self.azimuth),
            "elevation": None if self.elevation is None else float(self.elevation),
            "lookat": None if self.lookat is None else [float(value) for value in np.asarray(self.lookat, dtype=float)],
        }

    @classmethod
    def from_camera(cls, camera: object) -> "CameraSettings":
        return cls(
            distance=getattr(camera, "distance", None),
            azimuth=getattr(camera, "azimuth", None),
            elevation=getattr(camera, "elevation", None),
            lookat=np.asarray(getattr(camera, "lookat", np.zeros(3)), dtype=float).copy(),
        )


@dataclass(slots=True)
class PlaybackConfig:
    """Playback timing behaviour."""

    hz: float = 240.0
    slow: float = 1.0
    loop: bool = False

    def __post_init__(self) -> None:
        if self.hz <= 0:
            raise ValueError(f"hz must be positive. Got {self.hz}.")
        if self.slow <= 0:
            raise ValueError(f"slow must be positive. Got {self.slow}.")

    @property
    def step_dt(self) -> float:
        return self.slow / self.hz


@dataclass(slots=True)
class RecordingConfig:
    """Recording settings for backends that support offscreen rendering."""

    path: str | Path
    fps: float | None = None
    size: tuple[int, int] | None = None
    frames_dir: str | Path | None = None

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        if self.fps is not None and self.fps <= 0:
            raise ValueError(f"fps must be positive. Got {self.fps}.")
        if self.size is not None:
            width, height = self.size
            if width <= 0 or height <= 0:
                raise ValueError(f"size must contain positive integers. Got {self.size}.")
            self.size = (int(width), int(height))
        if self.frames_dir is not None:
            self.frames_dir = Path(self.frames_dir)


def coerce_playback_config(
    playback: PlaybackConfig | None = None,
    *,
    hz: float = 240.0,
    slow: float = 1.0,
    loop: bool = False,
) -> PlaybackConfig:
    """Build or validate a :class:`PlaybackConfig`."""

    if playback is None:
        return PlaybackConfig(hz=hz, slow=slow, loop=loop)
    if not isinstance(playback, PlaybackConfig):
        raise TypeError(f"Unsupported playback configuration: {type(playback)!r}")
    return playback


def coerce_camera_settings(
    camera: object | str | Path | Mapping[str, object] | None,
) -> CameraSettings | None:
    """Normalize camera-like input into :class:`CameraSettings`."""

    if camera is None:
        return None
    if isinstance(camera, (str, Path)):
        return load_camera_settings(camera)
    if isinstance(camera, CameraSettings):
        return camera
    if isinstance(camera, Mapping):
        return CameraSettings(
            distance=camera.get("distance"),
            azimuth=camera.get("azimuth"),
            elevation=camera.get("elevation"),
            lookat=camera.get("lookat"),
        )
    return CameraSettings.from_camera(camera)


def coerce_recording_config(
    recording: RecordingConfig | None = None,
    *,
    record_path: str | Path | None = None,
    record_fps: float | None = None,
    record_size: tuple[int, int] | None = None,
    record_frames_dir: str | Path | None = None,
) -> RecordingConfig | None:
    """Build or validate a :class:`RecordingConfig`."""

    if recording is not None:
        if (
            record_path is not None
            or record_fps is not None
            or record_size is not None
            or record_frames_dir is not None
        ):
            raise ValueError("Specify either recording or record_path/record_fps/record_size/record_frames_dir.")
        if not isinstance(recording, RecordingConfig):
            raise TypeError(f"Unsupported recording configuration: {type(recording)!r}")
        return recording

    if record_path is None:
        if record_frames_dir is not None:
            raise ValueError("record_frames_dir requires record_path.")
        return None

    return RecordingConfig(
        path=record_path,
        fps=record_fps,
        size=record_size,
        frames_dir=record_frames_dir,
    )


def normalize_runtime_config(
    *,
    playback: PlaybackConfig | None = None,
    camera: object | Mapping[str, object] | None = None,
    recording: RecordingConfig | None = None,
    hz: float = 240.0,
    slow: float = 1.0,
    loop: bool = False,
    record_path: str | Path | None = None,
    record_fps: float | None = None,
    record_size: tuple[int, int] | None = None,
    record_frames_dir: str | Path | None = None,
) -> tuple[PlaybackConfig, CameraSettings | None, RecordingConfig | None]:
    """Normalize loose runtime arguments into config objects."""

    return (
        coerce_playback_config(playback, hz=hz, slow=slow, loop=loop),
        coerce_camera_settings(camera),
        coerce_recording_config(
            recording,
            record_path=record_path,
            record_fps=record_fps,
            record_size=record_size,
            record_frames_dir=record_frames_dir,
        ),
    )


def _camera_settings_from_world_offset(
    world_offset: np.ndarray,
) -> CameraSettings:
    offset = np.asarray(world_offset, dtype=float).reshape(3)
    distance = float(np.linalg.norm(offset))
    if distance <= 1e-12:
        return CameraSettings(distance=0.0, azimuth=0.0, elevation=0.0, lookat=np.zeros(3, dtype=float))

    azimuth = float(np.degrees(np.arctan2(-offset[1], -offset[0])))
    elevation = float(np.degrees(np.arctan2(-offset[2], np.hypot(offset[0], offset[1]))))
    return CameraSettings(distance=distance, azimuth=azimuth, elevation=elevation, lookat=np.zeros(3, dtype=float))


def _rotation_x(theta: float) -> np.ndarray:
    cos_theta = float(math.cos(theta))
    sin_theta = float(math.sin(theta))
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cos_theta, -sin_theta],
            [0.0, sin_theta, cos_theta],
        ],
        dtype=float,
    )


_MESHCAT_ROTATED_TO_WORLD = _rotation_x(math.pi / 2.0)


def _coerce_threejs_matrix(data: Any) -> np.ndarray:
    matrix = np.asarray(data, dtype=float)
    if matrix.shape == (16,):
        return matrix.reshape(4, 4).T
    if matrix.shape == (4, 4):
        return matrix
    raise ValueError(f"Expected a Three.js matrix with shape (16,) or (4, 4). Got {matrix.shape}.")


def _coerce_threejs_translation(node: Mapping[str, Any]) -> np.ndarray:
    matrix = node.get("matrix")
    if matrix is not None:
        return _coerce_threejs_matrix(matrix)[:3, 3].copy()

    position = node.get("position")
    if position is None:
        return np.zeros(3, dtype=float)

    translation = np.asarray(position, dtype=float)
    if translation.shape != (3,):
        raise ValueError(f"Expected a Three.js position with shape (3,). Got {translation.shape}.")
    return translation.copy()


def _find_named_scene_child(node: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    children = node.get("children", ())
    if not isinstance(children, list):
        raise ValueError("Invalid scene JSON: 'children' must be a list.")

    for child in children:
        if isinstance(child, Mapping) and child.get("name") == name:
            return child
    raise ValueError(f"Could not find MeshCat scene node {name!r}.")


def _extract_meshcat_camera_settings(scene: Mapping[str, Any]) -> CameraSettings:
    object_node = scene.get("object", scene)
    if not isinstance(object_node, Mapping):
        raise ValueError("Invalid MeshCat scene JSON: missing root object.")

    camera_root = _find_named_scene_child(
        _find_named_scene_child(object_node, "Cameras"),
        "default",
    )
    rotated_node = _find_named_scene_child(camera_root, "rotated")
    camera_node = _find_named_scene_child(rotated_node, "<object>")

    lookat = _coerce_threejs_translation(camera_root)
    rotated_offset = _coerce_threejs_translation(camera_node)
    world_offset = _MESHCAT_ROTATED_TO_WORLD @ rotated_offset
    settings = _camera_settings_from_world_offset(world_offset)
    settings.lookat = lookat
    return settings


def load_camera_settings(path: str | Path) -> CameraSettings:
    """Load camera settings from an ``ei-vo`` camera JSON or a MeshCat ``scene.json``."""

    resolved_path = Path(path)
    data = json.loads(resolved_path.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError(f"Camera settings in {resolved_path} must decode to a JSON object.")

    if "camera" in data:
        camera_section = data["camera"]
        if not isinstance(camera_section, Mapping):
            raise ValueError(f"'camera' in {resolved_path} must be a JSON object.")
        return coerce_camera_settings(camera_section)  # type: ignore[return-value]

    if any(key in data for key in ("distance", "azimuth", "elevation", "lookat")):
        return coerce_camera_settings(data)  # type: ignore[return-value]

    return _extract_meshcat_camera_settings(data)


def save_camera_settings(
    camera: object | str | Path | Mapping[str, object],
    path: str | Path,
) -> Path:
    """Persist camera settings into a portable JSON preset file."""

    settings = coerce_camera_settings(camera)
    if settings is None:
        raise ValueError("camera must not be None.")

    resolved_path = Path(path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_path.write_text(f"{json.dumps(settings.to_dict(), indent=2)}\n", encoding="utf-8")
    return resolved_path


def resolve_recording_fps(
    playback: PlaybackConfig,
    recording: RecordingConfig,
) -> float:
    """Resolve the effective recording frame rate for a playback session."""

    fps = recording.fps if recording.fps is not None else (playback.hz / playback.slow)
    if fps <= 0:
        raise ValueError(f"fps must be positive. Got {fps}.")
    return float(fps)


__all__ = [
    "CameraSettings",
    "PlaybackConfig",
    "RecordingConfig",
    "coerce_playback_config",
    "coerce_camera_settings",
    "coerce_recording_config",
    "load_camera_settings",
    "normalize_runtime_config",
    "resolve_recording_fps",
    "save_camera_settings",
]
