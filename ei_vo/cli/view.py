"""Static model viewer CLI with MeshCat as the default renderer."""

from __future__ import annotations

import argparse
from typing import Sequence

import numpy as np

from ..core import Trajectory
from ..modeling import compute_link_poses, load_urdf_scene
from ..programs import default_waypoints
from . import playback

_VIEW_DEFAULTS = {
    "hz": 240.0,
    "loop": False,
    "slow": 1.0,
}
_VIEW_ONLY_REMOVED_DESTINATIONS = {
    "deg",
    "hz",
    "loop",
    "program",
    "segT",
    "slow",
    "trajectries",
}
_VIEW_ONLY_REMOVED_OPTIONS = {
    "--deg",
    "--demo",
    "--hz",
    "--loop",
    "--program",
    "--segT",
    "--slow",
    "--trajectries",
}


def _remove_actions(parser: argparse.ArgumentParser, *, destinations: set[str]) -> None:
    removed_actions = {action for action in parser._actions if action.dest in destinations}
    if not removed_actions:
        return

    parser._actions = [action for action in parser._actions if action not in removed_actions]
    for action in removed_actions:
        for option_string in action.option_strings:
            parser._option_string_actions.pop(option_string, None)

    for group in parser._action_groups:
        group._group_actions = [action for action in group._group_actions if action not in removed_actions]

    for group in parser._mutually_exclusive_groups:
        group._group_actions = [action for action in group._group_actions if action not in removed_actions]


def _load_view_scene(args: argparse.Namespace):
    playback._resolve_model_dof(args, trajectory_dof=None)
    return load_urdf_scene(args.model)


def _build_preview_row(scene) -> np.ndarray:
    preview = default_waypoints(scene.dof)[0]
    return scene.clamp(preview[None, :])[0]


def _visual_half_extents(visual) -> np.ndarray | None:
    if visual.geometry_type == "box" and visual.size is not None:
        return 0.5 * np.asarray(visual.size, dtype=float)
    if visual.geometry_type == "cylinder" and visual.radius is not None and visual.length is not None:
        return np.array(
            [float(visual.radius), float(visual.radius), 0.5 * float(visual.length)],
            dtype=float,
        )
    if visual.geometry_type == "sphere" and visual.radius is not None:
        radius = float(visual.radius)
        return np.array([radius, radius, radius], dtype=float)
    return None


def _scene_bounds(scene, row: np.ndarray) -> np.ndarray:
    link_poses = compute_link_poses(scene, row)
    mins: list[np.ndarray] = []
    maxs: list[np.ndarray] = []

    for link_name, visuals in scene.link_visuals.items():
        link_pose = link_poses.get(link_name)
        if link_pose is None:
            continue
        for visual in visuals:
            pose = link_pose @ visual.origin
            half_extents = _visual_half_extents(visual)
            center = np.asarray(pose[:3, 3], dtype=float)
            if half_extents is None:
                mins.append(center)
                maxs.append(center)
                continue
            rotation = np.asarray(pose[:3, :3], dtype=float)
            extents = np.abs(rotation) @ half_extents
            mins.append(center - extents)
            maxs.append(center + extents)

    if not mins or not maxs:
        positions = np.vstack([np.asarray(pose[:3, 3], dtype=float) for pose in link_poses.values()])
        return np.vstack((np.min(positions, axis=0), np.max(positions, axis=0)))

    return np.vstack((np.min(mins, axis=0), np.max(maxs, axis=0)))


def build_view_trajectory(args: argparse.Namespace) -> Trajectory:
    scene = _load_view_scene(args)
    row = _build_preview_row(scene)
    return Trajectory.from_positions(
        row[None, :],
        meta={"mode": "view"},
    )


def build_view_camera(args: argparse.Namespace) -> dict[str, object]:
    scene = _load_view_scene(args)
    row = _build_preview_row(scene)
    bounds = _scene_bounds(scene, row)
    center = 0.5 * (bounds[0] + bounds[1])
    extents = bounds[1] - bounds[0]
    scene_scale = max(float(np.max(extents)), 0.25)
    return {
        "distance": scene_scale * 2.75,
        "lookat": tuple(float(value) for value in center),
    }


def _merge_camera_defaults(
    requested_camera: dict[str, object] | None,
    default_camera: dict[str, object],
) -> dict[str, object]:
    if requested_camera is None:
        return dict(default_camera)

    merged = dict(default_camera)
    for key, value in requested_camera.items():
        if value is not None:
            merged[key] = value
    return merged


def _reject_motion_options(parser: argparse.ArgumentParser, argv: Sequence[str] | None) -> None:
    if argv is None:
        return

    invalid = sorted(
        {
            token.split("=", 1)[0]
            for token in argv
            if token.split("=", 1)[0] in _VIEW_ONLY_REMOVED_OPTIONS
        }
    )
    if invalid:
        parser.error(
            "ei-vo-view is for static model inspection. "
            f"Unsupported option(s): {', '.join(invalid)}. "
            "Use ei-vo-play for trajectory playback."
        )


def build_parser() -> argparse.ArgumentParser:
    parser = playback.build_parser()
    parser.description = "View a robot model in MeshCat or Pyrender."
    _remove_actions(parser, destinations=_VIEW_ONLY_REMOVED_DESTINATIONS)
    parser.set_defaults(**_VIEW_DEFAULTS)
    for action in parser._actions:
        if action.dest == "renderer":
            action.default = "meshcat"
            break
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    _reject_motion_options(parser, argv)
    args = parser.parse_args(argv)
    record_path, auto_dir = playback._resolve_recording(args)

    if auto_dir is not None and record_path is not None:
        print(f"[ei-vo] saving output to {record_path}")

    trajectory = build_view_trajectory(args)
    default_camera = build_view_camera(args)
    play_kwargs = playback._build_play_kwargs(args, record_path=record_path)
    play_kwargs["camera"] = _merge_camera_defaults(play_kwargs["camera"], default_camera)
    if play_kwargs["renderer"] == "pyrender":
        play_kwargs["interactive"] = True
    elif play_kwargs["renderer"] == "meshcat" and record_path is None and args.saveCamera is None:
        play_kwargs["hold_open"] = True

    play_result = playback.play(args.model, trajectory, **play_kwargs)
    playback._save_resolved_camera(
        save_path=args.saveCamera,
        requested_camera=play_kwargs["camera"],
        play_result=play_result,
    )
    return 0


__all__ = ["build_parser", "build_view_camera", "build_view_trajectory", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
