"""Recording-related helpers shared across demos and utilities."""

from __future__ import annotations

import os
import pathlib
import shutil
import subprocess
import tempfile
import time
from typing import Optional, Tuple

import numpy as np


def resolve_record_destination(
    record_arg: Optional[os.PathLike | str],
    *,
    prefix: str = "demo_",
    suffix: str = ".mp4",
) -> Tuple[Optional[str], Optional[str]]:
    """Resolve a recording destination from a CLI-style ``--record`` argument.

    Parameters
    ----------
    record_arg:
        Value passed to ``--record``. ``None`` means "no recording". An empty
        string or a path ending with ``os.sep`` indicates that the caller wants
        to automatically generate a filename under the given directory (or the
        default ``./recordings`` directory when empty).
    prefix:
        Filename prefix to use when auto-generating a file.
    suffix:
        Filename suffix/extension to use when auto-generating a file.

    Returns
    -------
    tuple[str | None, str | None]
        A pair ``(record_path, auto_dir)``. ``record_path`` is ``None`` when no
        recording was requested. ``auto_dir`` is the directory that should be
        created by the caller when an auto-generated filename is used; it is
        ``None`` when the caller provided a concrete filename.
    """

    if record_arg is None:
        return None, None

    record_value = os.fspath(record_arg).strip()

    if record_value == "":
        base_dir = pathlib.Path.cwd() / "recordings"
    else:
        candidate = pathlib.Path(record_value)
        if record_value.endswith(os.sep):
            base_dir = candidate
        elif candidate.exists() and candidate.is_dir():
            base_dir = candidate
        else:
            return candidate.as_posix(), None

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"{prefix}{timestamp}{suffix}"
    record_path = (base_dir / filename).as_posix()
    return record_path, base_dir.as_posix()


def resolve_video_output_path(
    path: os.PathLike | str,
    *,
    default_suffix: str = ".mp4",
) -> pathlib.Path:
    """Normalize a video output path, adding a default suffix when needed."""

    output_path = pathlib.Path(path)
    if output_path.suffix == "":
        output_path = output_path.with_suffix(default_suffix)
    return output_path


def _frame_sequence_name(output_path: pathlib.Path) -> str:
    stem = output_path.stem.strip()
    if not stem:
        stem = "recording"
    safe = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in stem)
    safe = safe.strip("._")
    return safe or "recording"


def _frame_sequence_glob(extension: str) -> str:
    suffix = extension if extension.startswith(".") else f".{extension}"
    return f"[0-9][0-9][0-9][0-9][0-9][0-9][0-9]{suffix}"


def clear_frame_sequence(
    frame_dir: os.PathLike | str,
    *,
    extension: str,
) -> None:
    """Remove numbered frame files for the given extension from ``frame_dir``."""

    directory = pathlib.Path(frame_dir)
    for path in directory.glob(_frame_sequence_glob(extension)):
        if path.is_file():
            path.unlink()


def prepare_frame_directory(
    output_path: os.PathLike | str,
    *,
    frames_dir: os.PathLike | str | None = None,
    temp_prefix: str = "ei_vo_frames_",
) -> tuple[pathlib.Path, tempfile.TemporaryDirectory[str] | None]:
    """Resolve a directory that will hold numbered frame files."""

    output = resolve_video_output_path(output_path)
    if frames_dir is None:
        temp_dir = tempfile.TemporaryDirectory(prefix=temp_prefix)
        return pathlib.Path(temp_dir.name), temp_dir

    root = pathlib.Path(frames_dir)
    frame_dir = root / f"{_frame_sequence_name(output)}_frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    return frame_dir, None


def _resolve_executable_candidate(value: str) -> str | None:
    candidate = value.strip()
    if not candidate:
        return None

    resolved = shutil.which(candidate)
    if resolved is not None:
        return resolved

    path = pathlib.Path(candidate).expanduser()
    if path.is_file() and os.access(path, os.X_OK):
        return path.as_posix()
    return None


def find_ffmpeg_executable() -> str:
    """Locate the ``ffmpeg`` executable used for video export."""

    configured = os.environ.get("EI_VO_FFMPEG")
    if configured:
        resolved = _resolve_executable_candidate(configured)
        if resolved is not None:
            return resolved
        raise RuntimeError(
            f"EI_VO_FFMPEG is set to {configured!r}, but no executable was found there."
        )

    resolved = _resolve_executable_candidate("ffmpeg")
    if resolved is not None:
        return resolved

    raise RuntimeError(
        "Video export requires the 'ffmpeg' executable. Install ffmpeg or set EI_VO_FFMPEG "
        "to an executable path."
    )


def coerce_rgb_frame(frame) -> np.ndarray:
    """Normalize a frame-like object into an RGB ``uint8`` image."""

    image = np.asarray(frame)
    if image.ndim == 2:
        image = np.repeat(image[:, :, None], 3, axis=2)
    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError(f"Expected an image shaped like (H, W, 3/4). Got {image.shape}.")
    if image.shape[2] == 4:
        image = image[:, :, :3]
    if image.dtype != np.uint8:
        scale = 255.0 if np.issubdtype(image.dtype, np.floating) and float(np.max(image)) <= 1.0 else 1.0
        image = np.clip(image * scale, 0.0, 255.0).astype(np.uint8)
    return np.ascontiguousarray(image)


def write_rgb_frame(path: os.PathLike | str, frame) -> pathlib.Path:
    """Write an RGB frame to disk using the binary PPM format."""

    output_path = pathlib.Path(path)
    image = coerce_rgb_frame(frame)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = f"P6\n{image.shape[1]} {image.shape[0]}\n255\n".encode("ascii")
    output_path.write_bytes(header + image.tobytes())
    return output_path


def _ffmpeg_output_args(output_path: pathlib.Path) -> list[str]:
    suffix = output_path.suffix.lower()
    if suffix in ("", ".mp4", ".mov"):
        return [
            "-vf",
            "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
        ]
    if suffix == ".webm":
        return [
            "-vf",
            "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-c:v",
            "libvpx-vp9",
            "-pix_fmt",
            "yuv420p",
            "-b:v",
            "0",
            "-crf",
            "30",
        ]
    if suffix == ".gif":
        return [
            "-filter_complex",
            "split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse",
        ]
    return []


def export_frame_sequence_to_video(
    frame_dir: os.PathLike | str,
    output_path: os.PathLike | str,
    *,
    fps: float,
    extension: str = ".ppm",
    ffmpeg_path: str | None = None,
) -> pathlib.Path:
    """Encode a numbered frame sequence into a video with ``ffmpeg``."""

    if fps <= 0:
        raise ValueError(f"fps must be positive. Got {fps}.")

    output = resolve_video_output_path(output_path).expanduser()
    absolute_output = output.resolve(strict=False)
    absolute_output.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = ffmpeg_path or find_ffmpeg_executable()
    suffix = extension if extension.startswith(".") else f".{extension}"
    pattern = f"%07d{suffix}"
    command = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-start_number",
        "0",
        "-framerate",
        f"{float(fps):.12g}",
        "-i",
        pattern,
        *_ffmpeg_output_args(output),
        absolute_output.as_posix(),
    ]
    result = subprocess.run(
        command,
        cwd=pathlib.Path(frame_dir),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        details = result.stderr.strip() or result.stdout.strip() or "ffmpeg failed without output."
        raise RuntimeError(f"Video export failed while encoding {output.name}: {details}")
    return output


class FrameSequenceWriter:
    """Collect RGB frames on disk and encode them into a video on close."""

    def __init__(
        self,
        path: os.PathLike | str,
        *,
        fps: float,
        extension: str = ".ppm",
        frames_dir: os.PathLike | str | None = None,
        temp_prefix: str = "ei_vo_frames_",
    ) -> None:
        if fps <= 0:
            raise ValueError(f"fps must be positive. Got {fps}.")
        self.output_path = resolve_video_output_path(path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.fps = float(fps)
        self.extension = extension if extension.startswith(".") else f".{extension}"
        self.ffmpeg_path = find_ffmpeg_executable()
        self._frame_index = 0
        self.frame_dir, self._temp_dir = prepare_frame_directory(
            self.output_path,
            frames_dir=frames_dir,
            temp_prefix=temp_prefix,
        )
        clear_frame_sequence(self.frame_dir, extension=self.extension)
        self._closed = False

    def append_data(self, frame) -> pathlib.Path:
        if self._closed:
            raise RuntimeError("Cannot append frames after the writer has been closed.")
        frame_path = self.frame_dir / f"{self._frame_index:07d}{self.extension}"
        write_rgb_frame(frame_path, frame)
        self._frame_index += 1
        return frame_path

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if self._frame_index > 0:
                export_frame_sequence_to_video(
                    self.frame_dir,
                    self.output_path,
                    fps=self.fps,
                    extension=self.extension,
                    ffmpeg_path=self.ffmpeg_path,
                )
        finally:
            if self._temp_dir is not None:
                self._temp_dir.cleanup()


__all__ = [
    "FrameSequenceWriter",
    "coerce_rgb_frame",
    "clear_frame_sequence",
    "export_frame_sequence_to_video",
    "find_ffmpeg_executable",
    "prepare_frame_directory",
    "resolve_record_destination",
    "resolve_video_output_path",
    "write_rgb_frame",
]
