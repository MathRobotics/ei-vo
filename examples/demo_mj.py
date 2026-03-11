#!/usr/bin/env python3
"""Compatibility entrypoint for the canonical playback CLI."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:  # pragma: no cover - exercised by direct script execution.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from examples._bootstrap import ensure_repo_root_on_path
else:
    from ._bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()

try:
    import mujoco as mj
except ImportError:  # pragma: no cover - only exercised when MuJoCo is absent.
    mj = None

from ei_vo import play
from ei_vo.cli.playback import build_parser, main
from ei_vo.core import load_angles, quintic, resolve_record_destination
from ei_vo.demo import (
    build_demo_trajectory as _build_demo_trajectory,
    build_sine_demo as _build_sine_demo,
    demo_waypoints,
    generate_demo_trajectory,
)

try:
    from ei_vo.render import render_mj
except Exception:  # pragma: no cover - only exercised when MuJoCo is absent.
    render_mj = None

_resolve_record_destination = resolve_record_destination


def build_demo_trajectory(q_wp, seg_T: float, hz: float):
    """Backward-compatible waypoint demo helper."""

    return _build_demo_trajectory(q_wp, segment_duration=seg_T, hz=hz)


def build_sine_demo(dof: int, T_sec: float, hz: float):
    """Backward-compatible sine demo helper."""

    return _build_sine_demo(dof, duration=T_sec, hz=hz)


def _prepare_play_invocation(args, traj_obj):
    """Backward-compatible helper mirroring the stable ``ei_vo.play`` API."""

    call_args = [args.model]
    call_kwargs = {
        "traj": traj_obj,
        "slow": args.slow,
        "hz": args.hz,
        "loop": args.loop,
    }
    if getattr(args, "renderer", None) is not None:
        call_kwargs["renderer"] = args.renderer
    if getattr(args, "record", None) is not None:
        call_kwargs["record_path"] = args.record
        if getattr(args, "recordFps", None) is not None:
            call_kwargs["record_fps"] = args.recordFps
        if getattr(args, "recordSize", None) is not None:
            call_kwargs["record_size"] = tuple(args.recordSize)
    return call_args, call_kwargs


__all__ = [
    "_prepare_play_invocation",
    "_resolve_record_destination",
    "build_demo_trajectory",
    "build_parser",
    "build_sine_demo",
    "demo_waypoints",
    "generate_demo_trajectory",
    "load_angles",
    "main",
    "mj",
    "play",
    "quintic",
    "render_mj",
]


if __name__ == "__main__":
    raise SystemExit(main())
