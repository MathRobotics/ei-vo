"""Backend-agnostic playback and recording configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

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
    camera: object | Mapping[str, object] | None,
) -> CameraSettings | None:
    """Normalize camera-like input into :class:`CameraSettings`."""

    if camera is None:
        return None
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
    "normalize_runtime_config",
    "resolve_recording_fps",
]
