"""Blender-based offline rendering backend."""

from __future__ import annotations

import importlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import hashlib
from typing import Any

import numpy as np

from ..config import PlaybackConfig, RecordingConfig, coerce_camera_settings
from ..core import (
    Trajectory,
    clear_frame_sequence,
    export_frame_sequence_to_video,
    find_ffmpeg_executable,
    prepare_frame_directory,
)
from ..modeling import load_robot_model

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
_VIDEO_SUFFIXES = {".gif", ".mov", ".mp4", ".webm"}
_DEFAULT_RECORD_SIZE = (1280, 720)
_DEFAULT_ENGINE = "eevee"
_DEFAULT_SAMPLES = 32
_DEFAULT_DRIVER = "auto"
_SCENE_CACHE_VERSION = "3"


def _resolve_blender_executable_candidate(value: str | os.PathLike[str]) -> str | None:
    candidate = os.fspath(value).strip()
    if not candidate:
        return None

    resolved = shutil.which(candidate)
    if resolved is not None:
        return resolved

    path = pathlib.Path(candidate).expanduser()
    if path.suffix.lower() == ".app" and path.is_dir():
        macos_dir = path / "Contents" / "MacOS"
        preferred = macos_dir / "Blender"
        if preferred.is_file() and os.access(preferred, os.X_OK):
            return preferred.as_posix()
        if macos_dir.is_dir():
            for child in sorted(macos_dir.iterdir()):
                if child.is_file() and os.access(child, os.X_OK):
                    return child.as_posix()
        return None

    if path.is_file() and os.access(path, os.X_OK):
        return path.as_posix()
    return None


def find_blender_executable(configured: str | os.PathLike[str] | None = None) -> str:
    """Locate the Blender executable used for offline rendering."""

    if configured is not None:
        resolved = _resolve_blender_executable_candidate(configured)
        if resolved is not None:
            return resolved
        raise RuntimeError(
            f"Configured Blender executable {os.fspath(configured)!r} could not be resolved."
        )

    env_value = os.environ.get("EI_VO_BLENDER")
    if env_value:
        resolved = _resolve_blender_executable_candidate(env_value)
        if resolved is not None:
            return resolved
        raise RuntimeError(
            f"EI_VO_BLENDER is set to {env_value!r}, but no Blender executable was found there."
        )

    resolved = _resolve_blender_executable_candidate("blender")
    if resolved is not None:
        return resolved

    if sys.platform == "darwin":
        for candidate in (
            pathlib.Path("/Applications/Blender.app"),
            pathlib.Path.home() / "Applications" / "Blender.app",
        ):
            resolved = _resolve_blender_executable_candidate(candidate)
            if resolved is not None:
                return resolved

    raise RuntimeError(
        "Blender rendering requires the 'blender' executable. Install Blender, add it to PATH, "
        "or set EI_VO_BLENDER to the executable path or .app bundle."
    )


def _find_blender_executable_optional(configured: str | os.PathLike[str] | None = None) -> str | None:
    try:
        return find_blender_executable(configured)
    except RuntimeError:
        return None


def _resolve_record_output_path(record_path: str | pathlib.Path) -> tuple[pathlib.Path, str]:
    output_path = pathlib.Path(record_path)
    suffix = output_path.suffix.lower()
    if suffix in _VIDEO_SUFFIXES:
        return output_path, "video"
    if suffix in _IMAGE_SUFFIXES:
        return output_path, "image"
    if suffix == "":
        return output_path.with_suffix(".mp4"), "video"
    raise ValueError(
        "Unsupported Blender output suffix "
        f"{output_path.suffix!r}. Use one of {', '.join(sorted(_IMAGE_SUFFIXES | _VIDEO_SUFFIXES))}."
    )


def _resolve_image_format(output_path: pathlib.Path) -> str:
    suffix = output_path.suffix.lower()
    if suffix == ".png":
        return "PNG"
    if suffix in {".jpg", ".jpeg"}:
        return "JPEG"
    raise ValueError(f"Unsupported image output suffix for Blender: {output_path.suffix!r}.")


def _resolve_record_fps(*, playback: PlaybackConfig, record_fps: float | None) -> float:
    fps = float(record_fps) if record_fps is not None else (playback.hz / playback.slow)
    if fps <= 0:
        raise ValueError(f"record_fps must be positive. Got {record_fps}.")
    return fps


def _scene_cache_root() -> pathlib.Path:
    return pathlib.Path(tempfile.gettempdir()) / "ei_vo_blender_cache"


def _default_scene_cache_path(model_path: pathlib.Path) -> pathlib.Path:
    stat = model_path.stat()
    digest = hashlib.sha256()
    digest.update(_SCENE_CACHE_VERSION.encode("ascii"))
    digest.update(model_path.as_posix().encode("utf-8"))
    digest.update(str(stat.st_size).encode("ascii"))
    digest.update(str(stat.st_mtime_ns).encode("ascii"))
    filename = f"{model_path.stem}_{digest.hexdigest()[:16]}.blend"
    return _scene_cache_root() / filename


def _resolve_scene_cache_path(
    model_path: pathlib.Path,
    *,
    enabled: bool,
    scene_cache_path: str | pathlib.Path | None,
) -> pathlib.Path | None:
    if not enabled:
        return None
    if scene_cache_path is None:
        return _default_scene_cache_path(model_path)

    resolved = pathlib.Path(scene_cache_path).expanduser()
    if resolved.suffix.lower() != ".blend":
        resolved = resolved.with_suffix(".blend")
    return resolved


def _trajectory_source_times(trajectory: Trajectory, *, hz: float) -> np.ndarray:
    if trajectory.steps == 1:
        return np.array([0.0], dtype=float)
    if trajectory.t is not None:
        t = np.asarray(trajectory.t, dtype=float)
        if np.all(np.diff(t) >= 0.0):
            return t
    return np.arange(trajectory.steps, dtype=float) / float(hz)


def _resample_trajectory_for_recording(
    trajectory: Trajectory,
    *,
    playback: PlaybackConfig,
    record_fps: float,
) -> np.ndarray:
    original_positions = np.asarray(trajectory.q, dtype=float)
    if trajectory.steps <= 1:
        return original_positions

    source_times = _trajectory_source_times(trajectory, hz=playback.hz)
    source_duration = float(source_times[-1])
    if source_duration <= 0.0:
        return original_positions[:1]

    playback_duration = source_duration * playback.slow
    target_playback_times = np.arange(
        0.0,
        playback_duration + (0.5 / float(record_fps)),
        1.0 / float(record_fps),
        dtype=float,
    )
    if target_playback_times.size == 0 or target_playback_times[-1] < playback_duration:
        target_playback_times = np.append(target_playback_times, playback_duration)
    if target_playback_times.size >= trajectory.steps:
        return original_positions
    target_source_times = np.clip(target_playback_times / playback.slow, source_times[0], source_times[-1])

    resampled = np.empty((target_source_times.shape[0], trajectory.dof), dtype=float)
    for joint_index in range(trajectory.dof):
        resampled[:, joint_index] = np.interp(
            target_source_times,
            source_times,
            trajectory.q[:, joint_index],
        )
    return resampled


def _camera_manifest(camera: object) -> dict[str, Any] | None:
    settings = coerce_camera_settings(camera)
    if settings is None:
        return None
    return {
        "distance": None if settings.distance is None else float(settings.distance),
        "azimuth": None if settings.azimuth is None else float(settings.azimuth),
        "elevation": None if settings.elevation is None else float(settings.elevation),
        "lookat": None if settings.lookat is None else [float(value) for value in settings.lookat],
    }


def _blender_script_path() -> pathlib.Path:
    return pathlib.Path(__file__).with_name("render_blender_script.py")


def _run_blender_process(blender_executable: str, manifest_path: pathlib.Path) -> None:
    command = [
        blender_executable,
        "--background",
        "--factory-startup",
        "--python-exit-code",
        "1",
        "--python",
        _blender_script_path().as_posix(),
        "--",
        manifest_path.as_posix(),
    ]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    assert process.stdout is not None
    output_lines: list[str] = []
    for line in process.stdout:
        output_lines.append(line)
        print(f"[ei-vo:blender] {line.rstrip()}")

    return_code = process.wait()
    if return_code == 0:
        return

    details = "".join(output_lines).strip() or "Blender failed without output."
    raise RuntimeError(f"Blender render failed with exit code {return_code}: {details}")


def _import_bpy_renderer():
    try:
        bpy = importlib.import_module("bpy")
    except ImportError as exc:
        raise RuntimeError(
            "The Blender bpy fallback requires the optional Python package 'bpy'. "
            "Install it with uv add bpy in a compatible Python environment."
        ) from exc

    try:
        renderer_module = importlib.import_module("ei_vo.render.render_blender_script")
    except ImportError as exc:
        raise RuntimeError("Failed to import the shared bpy renderer module.") from exc
    return bpy, renderer_module


def _run_bpy_module(manifest: dict[str, Any], *, blender_executable: str | None) -> None:
    bpy, renderer_module = _import_bpy_renderer()
    previous_binary_path = getattr(getattr(bpy, "app", None), "binary_path", None)
    try:
        if blender_executable is not None and previous_binary_path is not None:
            bpy.app.binary_path = blender_executable
        print("[ei-vo:blender] rendering with bpy module")
        renderer_module.render_manifest(manifest)
    finally:
        if previous_binary_path is not None:
            bpy.app.binary_path = previous_binary_path


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
    blender_executable: str | pathlib.Path | None = None,
    driver: str = _DEFAULT_DRIVER,
    engine: str = _DEFAULT_ENGINE,
    samples: int = _DEFAULT_SAMPLES,
    floor: bool = True,
    scene_cache: bool = True,
    scene_cache_path: str | pathlib.Path | None = None,
    image_frame_index: int | None = None,
    debug_links_path: str | pathlib.Path | None = None,
    kinematics_backend: str | None = None,
    kinematics_model_path: str | pathlib.Path | None = None,
    base_link: str | None = None,
    end_link: str | None = None,
):
    """Render a URDF trajectory offline through Blender."""

    del loop, kinematics_backend, kinematics_model_path, base_link, end_link

    if model_path is None:
        raise ValueError("The 'blender' renderer requires model_path.")
    if record_path is None:
        raise ValueError("The 'blender' renderer requires record_path for offline output.")

    model = pathlib.Path(model_path).expanduser().resolve(strict=False)
    if model.suffix.lower() != ".urdf":
        raise ValueError(f"The 'blender' renderer only supports URDF models. Got {model.suffix!r}.")

    trajectory = Trajectory.coerce(traj)
    if trajectory.steps <= 0:
        raise ValueError("Trajectory must contain at least one frame.")

    robot = load_robot_model(model)
    if trajectory.dof != robot.dof:
        raise ValueError(
            f"Trajectory DOF ({trajectory.dof}) does not match model DOF ({robot.dof})."
        )

    normalized_engine = str(engine).strip().lower()
    if not normalized_engine:
        raise ValueError("engine must not be empty.")
    if normalized_engine not in {"cycles", "eevee", "workbench"}:
        raise ValueError(f"Unsupported Blender engine {engine!r}.")
    normalized_driver = str(driver).strip().lower()
    if normalized_driver not in {"auto", "bpy", "subprocess"}:
        raise ValueError(f"Unsupported Blender driver {driver!r}.")
    if int(samples) <= 0:
        raise ValueError(f"samples must be positive. Got {samples}.")

    output_path, output_mode = _resolve_record_output_path(record_path)
    recording = RecordingConfig(
        path=output_path,
        fps=record_fps,
        size=record_size,
        frames_dir=record_frames_dir,
    )
    if output_mode == "image" and record_frames_dir is not None:
        raise ValueError("record_frames_dir is only supported for Blender video output.")
    if output_mode == "video" and image_frame_index is not None:
        raise ValueError("image_frame_index is only supported for Blender image output.")
    if output_mode == "video" and debug_links_path is not None:
        raise ValueError("debug_links_path is only supported for Blender image output.")

    playback = PlaybackConfig(hz=hz, slow=slow)
    render_size = recording.size or _DEFAULT_RECORD_SIZE
    blender = None if normalized_driver == "bpy" else _find_blender_executable_optional(blender_executable)

    ffmpeg_path = None
    frame_dir: pathlib.Path | None = None
    temp_frames_dir = None
    cache_path = _resolve_scene_cache_path(
        model,
        enabled=bool(scene_cache),
        scene_cache_path=scene_cache_path,
    )
    try:
        output_spec: dict[str, Any]
        render_positions = np.asarray(trajectory.q, dtype=float)
        if output_mode == "video":
            ffmpeg_path = find_ffmpeg_executable()
            fps = _resolve_record_fps(playback=playback, record_fps=record_fps)
            render_positions = _resample_trajectory_for_recording(
                trajectory,
                playback=playback,
                record_fps=fps,
            )
            frame_dir, temp_frames_dir = prepare_frame_directory(
                recording.path,
                frames_dir=recording.frames_dir,
                temp_prefix="ei_vo_blender_frames_",
            )
            frame_dir = frame_dir.resolve(strict=False)
            clear_frame_sequence(frame_dir, extension=".png")
            output_spec = {
                "kind": "video",
                "frame_dir": frame_dir.as_posix(),
                "fps": float(fps),
            }
        else:
            image_output = recording.path.expanduser().resolve(strict=False)
            image_output.parent.mkdir(parents=True, exist_ok=True)
            frame_index = trajectory.steps - 1 if image_frame_index is None else int(image_frame_index)
            if frame_index < 0 or frame_index >= trajectory.steps:
                raise ValueError(
                    f"image_frame_index must be in [0, {trajectory.steps - 1}] for this trajectory. "
                    f"Got {image_frame_index}."
                )
            output_spec = {
                "kind": "image",
                "path": image_output.as_posix(),
                "format": _resolve_image_format(image_output),
                "frame_index": frame_index,
            }
        debug_spec = None
        if debug_links_path is not None:
            debug_output = pathlib.Path(debug_links_path).expanduser().resolve(strict=False)
            debug_output.parent.mkdir(parents=True, exist_ok=True)
            debug_spec = {
                "links_path": debug_output.as_posix(),
            }

        manifest = {
            "model_path": model.as_posix(),
            "trajectory": [[float(value) for value in row] for row in render_positions],
            "camera": _camera_manifest(camera),
            "scene_cache": None if cache_path is None else {"path": cache_path.as_posix()},
            "debug": debug_spec,
            "render": {
                "width": int(render_size[0]),
                "height": int(render_size[1]),
                "engine": normalized_engine,
                "samples": int(samples),
                "floor": bool(floor),
            },
            "output": output_spec,
        }

        render_failures: list[str] = []
        rendered = False
        if normalized_driver in {"auto", "subprocess"}:
            if blender is None:
                render_failures.append(
                    "subprocess backend unavailable because no Blender executable was found"
                )
            else:
                try:
                    with tempfile.TemporaryDirectory(prefix="ei_vo_blender_manifest_") as manifest_dir:
                        manifest_path = pathlib.Path(manifest_dir) / "scene.json"
                        manifest_path.write_text(
                            json.dumps(manifest, separators=(",", ":")),
                            encoding="utf-8",
                        )
                        _run_blender_process(blender, manifest_path)
                    rendered = True
                except RuntimeError as exc:
                    render_failures.append(str(exc))
                    if normalized_driver == "auto":
                        print(f"[ei-vo:blender] subprocess backend failed, retrying with bpy: {exc}")

        if not rendered and normalized_driver in {"auto", "bpy"}:
            try:
                _run_bpy_module(manifest, blender_executable=blender)
                rendered = True
            except RuntimeError as exc:
                render_failures.append(str(exc))

        if not rendered:
            details = " | ".join(render_failures) if render_failures else "no render backend was attempted"
            raise RuntimeError(f"Blender render failed: {details}")

        if output_mode == "video":
            assert frame_dir is not None
            if not any(frame_dir.glob("*.png")):
                raise RuntimeError("Blender completed without producing any PNG frames.")
            return export_frame_sequence_to_video(
                frame_dir,
                recording.path,
                fps=output_spec["fps"],
                extension=".png",
                ffmpeg_path=ffmpeg_path,
            )

        image_output = pathlib.Path(output_spec["path"])
        if not image_output.exists():
            raise RuntimeError("Blender completed without producing the requested image.")
        return image_output
    finally:
        if temp_frames_dir is not None:
            temp_frames_dir.cleanup()


__all__ = ["find_blender_executable", "play"]
