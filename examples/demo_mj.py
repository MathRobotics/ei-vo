#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import inspect
import math
import os
import pathlib
import sys
import time
import types
import warnings

import numpy as np
import mujoco as mj

from ei_vo import play
from ei_vo.core import load_angles, quintic
from ei_vo.render import render_mj

# ---------------------------
# Demo trajectory generation (when no angle file is provided)
# ---------------------------
def demo_waypoints(dof: int) -> np.ndarray:
    """Return conservative example poses for ``dof`` joints (rows=poses, cols=dof)."""

    base = np.linspace(-0.6, 0.6, dof, dtype=float)
    phase = np.linspace(0.0, math.pi, dof, dtype=float)

    offsets = [
        np.zeros(dof, dtype=float),
        0.35 * np.sin(phase),
        -0.25 * np.sin(phase + math.pi / 4.0),
        0.30 * np.sin(phase + math.pi / 2.0),
        np.zeros(dof, dtype=float),
    ]

    poses = [base + off for off in offsets]
    poses[-1] = poses[0].copy()  # Return to the starting configuration
    return np.vstack(poses)

def build_demo_trajectory(q_wp: np.ndarray, seg_T: float, hz: float) -> np.ndarray:
    """Connect waypoint pairs with quintic curves and concatenate the segments."""
    dt = 1.0 / max(hz, 1e-6)
    chunks = []
    for i in range(len(q_wp)-1):
        chunks.append(quintic(q_wp[i], q_wp[i+1], seg_T, dt)[:-1])  # Drop overlap at the boundary
    chunks.append(q_wp[-1][None, :])
    return np.vstack(chunks)

def build_sine_demo(dof: int, T_sec: float, hz: float) -> np.ndarray:
    """Simple sinusoidal demo within a conservative range (rows=T*hz, cols=dof)."""
    dt = 1.0 / max(hz, 1e-6)
    t = np.arange(0.0, T_sec + 1e-12, dt)
    T = t.shape[0]
    q = np.zeros((T, dof), dtype=float)
    base = np.linspace(-0.6, 0.6, dof, dtype=float)
    amp = np.linspace(0.15, 0.30, dof, dtype=float)
    freq = np.linspace(0.20, 0.35, dof, dtype=float)  # Hz
    phase = np.linspace(0.0, math.pi, dof, dtype=float)
    for i in range(dof):
        q[:, i] = base[i] + amp[i] * np.sin(2 * math.pi * freq[i] * t + phase[i])
    return q

# ---------------------------
# Main
# ---------------------------
def _resolve_record_destination(record_arg):
    """Resolve the desired recording path.

    Returns a tuple of ``(path_or_none, auto_dir)`` where ``auto_dir`` is the
    directory that was auto-created when the caller did not supply an explicit
    filename. ``auto_dir`` is ``None`` when the caller provided a concrete
    destination.
    """

    if record_arg is None:
        return None, None

    record_value = os.fspath(record_arg)
    record_value = record_value.strip()

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
    filename = f"demo_{timestamp}.mp4"
    record_path = (base_dir / filename).as_posix()
    return record_path, base_dir.as_posix()


def _prepare_play_invocation(args, traj_obj):
    """Build argument lists for ``ei.play`` depending on its signature."""

    sig = inspect.signature(play)
    params = sig.parameters

    call_args = [args.model]
    call_kwargs = {}

    if "traj" in params:
        call_kwargs["traj"] = traj_obj
    else:
        call_args.append(traj_obj)

    if "slow" in params:
        call_kwargs["slow"] = args.slow
    if "hz" in params:
        call_kwargs["hz"] = args.hz
    if "loop" in params:
        call_kwargs["loop"] = args.loop

    record_requested = args.record is not None
    record_supported = "record_path" in params
    fps_supported = "record_fps" in params
    size_supported = "record_size" in params

    if record_requested and record_supported:
        call_kwargs["record_path"] = args.record
        if fps_supported and args.recordFps is not None:
            call_kwargs["record_fps"] = args.recordFps
        if size_supported and args.recordSize:
            call_kwargs["record_size"] = tuple(args.recordSize)
        if not fps_supported and args.recordFps is not None:
            warnings.warn(
                "recordFps specified but ei.play() does not accept a record_fps argument; ignoring.",
                RuntimeWarning,
            )
        if not size_supported and args.recordSize:
            warnings.warn(
                "recordSize specified but ei.play() does not accept a record_size argument; ignoring.",
                RuntimeWarning,
            )
    else:
        if record_requested:
            warnings.warn(
                "Recording was requested but the installed ei.play() does not accept a 'record_path' argument.",
                RuntimeWarning,
            )
        if args.recordFps is not None:
            warnings.warn(
                "recordFps specified without recording support; ignoring.",
                RuntimeWarning,
            )
        if args.recordSize:
            warnings.warn(
                "recordSize specified without recording support; ignoring.",
                RuntimeWarning,
            )

    return call_args, call_kwargs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to the Panda MJCF (panda.xml)")
    ap.add_argument("--angles", default=None, help="Joint angle file CSV/NPY/JSON, shape=(T, DOF) (optional)")
    ap.add_argument("--deg", action="store_true", help="Specify when the angle file is in degrees")
    ap.add_argument("--hz", type=float, default=240.0, help="Playback frequency [Hz] (demo/file)")
    ap.add_argument("--loop", action="store_true", help="Loop playback at the end")
    ap.add_argument("--demo", choices=["wp", "sine"], default="wp", help="Demo type when no angle file: wp (waypoints) / sine")
    ap.add_argument("--segT", type=float, default=1.5, help="Segment duration [s] for wp demo")
    ap.add_argument("--slow", type=float, default=1.0, help="Slowdown multiplier (>1 plays slower)")
    ap.add_argument(
        "--record",
        nargs="?",
        const="",
        default=None,
        help="Path to save the recording (e.g., output.mp4). Defaults to recordings/ when omitted or directory-only",
    )
    ap.add_argument("--recordFps", type=float, default=None, help="Recording frame rate [fps] (default: playback fps)")
    ap.add_argument("--recordSize", type=int, nargs=2, metavar=("W", "H"), default=None,
                    help="Recording width[px] and height[px] (default: 1280x720)")

    args = ap.parse_args()

    if not os.path.isfile(args.model):
        raise FileNotFoundError(args.model)

    mj_model = mj.MjModel.from_xml_path(args.model)
    model_dof = len(render_mj.detect_arm_joint_qaddr(mj_model))
    if model_dof == 0:
        raise RuntimeError("Failed to identify arm joints from the model")

    record_path, auto_dir = _resolve_record_destination(args.record)
    args.record = record_path

    if auto_dir is not None and record_path is not None:
        print(f"[demo_mj] --record was given without a filename; saving to {record_path}")

    # Prepare trajectory
    if args.angles is None:
        if args.demo == "wp":
            q_wp = demo_waypoints(model_dof)
            q = build_demo_trajectory(q_wp, seg_T=args.segT, hz=args.hz)
        else:
            q = build_sine_demo(model_dof, T_sec=10.0, hz=args.hz)
    else:
        q = load_angles(args.angles, deg=args.deg)
        if q.shape[1] != model_dof:
            raise ValueError(
                f"Number of angle columns ({q.shape[1]}) does not match model DOF ({model_dof})"
            )

    traj_obj = types.SimpleNamespace(q=q)
    call_args, call_kwargs = _prepare_play_invocation(args, traj_obj)
    play(*call_args, **call_kwargs)

if __name__ == "__main__":
    main()
