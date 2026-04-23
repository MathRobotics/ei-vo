"""Canonical CLI entrypoint for playback and rendering."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Sequence

from .. import play
from ..backends import KinematicsSpec
from ..config import CameraSettings, coerce_camera_settings, save_camera_settings
from ..core import Trajectory, resolve_record_destination
from ..kinematics import available_kinematics_backends
from ..modeling import load_robot_model
from ..programs import available_programs, normalize_program_mode
from ..render import available_renderers
from ..workflows import trajectory_from_file, trajectory_from_program


def _parse_program_mode(value: str) -> str:
    try:
        return normalize_program_mode(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replay or generate robot joint trajectories.")
    parser.add_argument(
        "--model",
        default=None,
        help="Path to the URDF model file (required for the built-in renderers)",
    )
    parser.add_argument(
        "--trajectries",
        default=None,
        help="Trajectory file in CSV / NPY / JSON format with shape=(T, DOF)",
    )
    parser.add_argument("--deg", action="store_true", help="Interpret --trajectries input as degrees")
    parser.add_argument("--hz", type=float, default=240.0, help="Playback frequency [Hz]")
    parser.add_argument(
        "--renderer",
        choices=available_renderers(),
        default="matplotlib",
        help="Renderer backend to use",
    )
    parser.add_argument(
        "--backend",
        choices=available_kinematics_backends(),
        default=None,
        help="Optional kinematics backend to attach to the playback workflow",
    )
    parser.add_argument("--base-link", default=None, help="Base link for --backend")
    parser.add_argument("--end-link", default=None, help="End link for --backend")
    parser.add_argument("--loop", action="store_true", help="Loop playback until the viewer is closed")
    parser.add_argument(
        "--program",
        type=_parse_program_mode,
        choices=available_programs(),
        default=argparse.SUPPRESS,
        help="Built-in motion program to use when --trajectries is omitted",
    )
    parser.add_argument(
        "--demo",
        dest="program",
        type=_parse_program_mode,
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--segT", type=float, default=1.5, help="Waypoint segment duration [s]")
    parser.add_argument("--slow", type=float, default=1.0, help="Playback slowdown factor (>1 is slower)")
    parser.add_argument("--cameraDistance", type=float, default=None, help="Camera distance")
    parser.add_argument("--cameraAzimuth", type=float, default=None, help="Camera azimuth [deg]")
    parser.add_argument("--cameraElevation", type=float, default=None, help="Camera elevation [deg]")
    parser.add_argument(
        "--cameraFile",
        default=None,
        help="Camera preset JSON path or MeshCat scene.json to reuse",
    )
    parser.add_argument(
        "--cameraLookat",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=None,
        help="Camera look-at point",
    )
    parser.add_argument(
        "--saveCamera",
        default=None,
        help="Write the resolved camera settings to a reusable JSON preset",
    )
    parser.add_argument(
        "--record",
        nargs="?",
        const="",
        default=None,
        help="Recording output path. Omitting the filename writes into ./recordings/",
    )
    parser.add_argument(
        "--recordFps",
        type=float,
        default=None,
        help="Recording frame rate [fps] for video-capable renderers",
    )
    parser.add_argument(
        "--recordSize",
        type=int,
        nargs=2,
        metavar=("W", "H"),
        default=None,
        help="Recording width and height in pixels for video-capable renderers",
    )
    parser.add_argument(
        "--recordFramesDir",
        default=None,
        help="Directory root used to persist numbered recording frames alongside video export",
    )
    return parser


def _load_model_dof(model_path: str) -> int:
    return load_robot_model(model_path).dof


def _require_urdf_model_path(model_path: str) -> str:
    if Path(model_path).suffix.lower() != ".urdf":
        raise ValueError(f"Only URDF models are supported. Got {model_path!r}.")
    return model_path


def _resolve_model_dof(args: argparse.Namespace, *, trajectory_dof: int | None) -> int:
    if args.model is not None:
        _require_urdf_model_path(args.model)
        if not os.path.isfile(args.model):
            raise FileNotFoundError(args.model)
        return _load_model_dof(args.model)

    if args.renderer in {"matplotlib", "meshcat", "pyrender"}:
        raise ValueError(f"--model is required when using the {args.renderer} renderer.")

    if trajectory_dof is not None:
        return trajectory_dof
    raise ValueError("Specify --model when using the built-in renderers.")


def build_trajectory(args: argparse.Namespace) -> Trajectory:
    if args.trajectries is not None:
        trajectory = trajectory_from_file(args.trajectries, deg=args.deg, hz=args.hz)
        model_dof = _resolve_model_dof(args, trajectory_dof=trajectory.dof)
        if trajectory.dof != model_dof:
            raise ValueError(
                f"Number of trajectory columns ({trajectory.dof}) does not match model DOF ({model_dof})"
            )
        return trajectory

    dof = _resolve_model_dof(args, trajectory_dof=None)
    program = getattr(args, "program", "waypoints")
    return trajectory_from_program(
        dof,
        program=program,
        hz=args.hz,
        segment_duration=args.segT,
        meta={"program": program},
    )


def _resolve_recording(args: argparse.Namespace) -> tuple[str | None, str | None]:
    artifact_map = {
        "matplotlib": ("matplotlib_", ".png"),
        "meshcat": ("meshcat_", ".html"),
        "pyrender": ("pyrender_", ".mp4"),
    }
    artifact_prefix, artifact_suffix = artifact_map[args.renderer]
    return resolve_record_destination(
        args.record,
        prefix=artifact_prefix,
        suffix=artifact_suffix,
    )


def _build_kinematics_spec(args: argparse.Namespace) -> KinematicsSpec | None:
    if args.backend is None:
        if args.base_link is not None or args.end_link is not None:
            raise ValueError(
                "Specify --backend when using --base-link or --end-link."
            )
        return None

    model_path = args.model
    if model_path is not None:
        _require_urdf_model_path(model_path)
        if not os.path.isfile(model_path):
            raise FileNotFoundError(model_path)

    return KinematicsSpec(
        args.backend,
        model_path=model_path,
        base_link=args.base_link,
        end_link=args.end_link,
    )


def _camera_mapping(camera: object) -> dict[str, object]:
    settings = coerce_camera_settings(camera)
    if settings is None:
        raise ValueError("camera must not be None.")

    return {
        "distance": settings.distance,
        "azimuth": settings.azimuth,
        "elevation": settings.elevation,
        "lookat": None if settings.lookat is None else tuple(float(value) for value in settings.lookat),
    }


def _camera_overrides(args: argparse.Namespace) -> dict[str, object]:
    overrides: dict[str, object] = {}
    if args.cameraDistance is not None:
        overrides["distance"] = args.cameraDistance
    if args.cameraAzimuth is not None:
        overrides["azimuth"] = args.cameraAzimuth
    if args.cameraElevation is not None:
        overrides["elevation"] = args.cameraElevation
    if args.cameraLookat is not None:
        overrides["lookat"] = tuple(args.cameraLookat)
    return overrides


def _build_camera(args: argparse.Namespace) -> dict[str, object] | None:
    base_camera = None if args.cameraFile is None else _camera_mapping(args.cameraFile)
    overrides = _camera_overrides(args)
    if base_camera is None and not overrides:
        return None
    if base_camera is None:
        return overrides

    merged = dict(base_camera)
    merged.update(overrides)
    return merged


def _build_play_kwargs(args: argparse.Namespace, *, record_path: str | None) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "slow": args.slow,
        "hz": args.hz,
        "camera": _build_camera(args),
        "loop": args.loop,
        "record_path": record_path,
        "record_fps": args.recordFps,
        "record_size": tuple(args.recordSize) if args.recordSize is not None else None,
        "record_frames_dir": args.recordFramesDir,
        "renderer": args.renderer,
        "kinematics": _build_kinematics_spec(args),
    }
    return kwargs


def _save_resolved_camera(
    *,
    save_path: str | None,
    requested_camera: object,
    play_result: object,
) -> None:
    if save_path is None:
        return

    camera = play_result if isinstance(play_result, CameraSettings) else requested_camera
    if camera is None:
        raise ValueError("Specify --cameraFile or camera options when using --saveCamera.")

    saved_camera_path = save_camera_settings(camera, save_path)
    print(f"[ei-vo] saved camera preset to {saved_camera_path}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    record_path, auto_dir = _resolve_recording(args)

    if auto_dir is not None and record_path is not None:
        print(f"[ei-vo] saving output to {record_path}")

    trajectory = build_trajectory(args)
    play_kwargs = _build_play_kwargs(args, record_path=record_path)
    play_result = play(args.model, trajectory, **play_kwargs)
    _save_resolved_camera(
        save_path=args.saveCamera,
        requested_camera=play_kwargs["camera"],
        play_result=play_result,
    )
    return 0


__all__ = [
    "available_programs",
    "build_parser",
    "build_trajectory",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
