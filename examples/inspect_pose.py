#!/usr/bin/env python3
"""Interactively inspect a MuJoCo model without providing a trajectory."""
from __future__ import annotations

import argparse
from typing import Iterable, Optional

from ei_vo.render.pose_viewer import inspect_pose


def _parse_initial_pose(values: Optional[Iterable[str]]):
    if values is None:
        return None
    floats = [float(v) for v in values]
    return floats


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Launch an interactive viewer that lets you inspect a MuJoCo model and "
            "adjust joint angles without precomputed trajectories."
        )
    )
    parser.add_argument("model", help="Path to the MJCF model (XML) to visualise.")
    parser.add_argument(
        "--dof",
        type=int,
        default=None,
        help=(
            "Number of arm joints to control. By default all hinge joints except "
            "fingers/grippers are detected automatically."
        ),
    )
    parser.add_argument(
        "--step",
        type=float,
        default=5.0,
        help="Increment/decrement step in degrees for keyboard input (default: 5.0).",
    )
    parser.add_argument(
        "--pose",
        nargs="+",
        type=str,
        metavar="ANGLE",
        help="Initial joint angles in radians (one value per controlled joint).",
    )
    parser.add_argument(
        "--no-clamp",
        action="store_true",
        help="Disable automatic clamping to the joint limits defined in the model.",
    )
    args = parser.parse_args(argv)

    initial_pose = _parse_initial_pose(args.pose)

    inspect_pose(
        args.model,
        expected_dof=args.dof,
        initial_pose=initial_pose,
        step_deg=args.step,
        clamp=not args.no_clamp,
    )


if __name__ == "__main__":
    main()
