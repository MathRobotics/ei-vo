"""Launch the standard MuJoCo viewer for quickly inspecting a model pose.

This script intentionally keeps the experience close to the default MuJoCo GUI.
It loads an MJCF, optionally applies an initial joint configuration, and hands
control over to the viewer so that you can use the built-in mouse and keyboard
shortcuts (e.g. the joint, control, and sensor panels) to explore the model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import mujoco
import numpy as np


def _load_pose(path: Path, dof: int) -> np.ndarray:
    """Load a qpos vector from ``path`` supporting JSON, NPY, and CSV files."""

    suffix = path.suffix.lower()
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        pose = np.array(payload, dtype=float)
    elif suffix == ".npy":
        pose = np.load(path)
    else:  # treat anything else as CSV / text
        pose = np.loadtxt(path, delimiter=",")

    pose = np.asarray(pose, dtype=float)
    if pose.ndim == 2:
        if pose.shape[0] == 1:
            pose = pose[0]
        else:
            raise ValueError(
                f"Expected a 1D pose, got array with shape {pose.shape}. "
                "If you have a trajectory, pick a single row to load."
            )
    if pose.shape[0] != dof:
        raise ValueError(f"Pose length {pose.shape[0]} does not match model DOF {dof}.")
    return pose


def _parse_pose(values: Sequence[float] | None, dof: int) -> np.ndarray | None:
    if values is None:
        return None
    pose = np.asarray(values, dtype=float)
    if pose.shape[0] != dof:
        raise ValueError(
            f"Expected {dof} values for --pose, but received {pose.shape[0]} entries."
        )
    return pose


def inspect_pose(model: mujoco.MjModel, pose: np.ndarray | None) -> None:
    data = mujoco.MjData(model)

    if pose is not None:
        data.qpos[:] = pose
        mujoco.mj_forward(model, data)

    print("Launching MuJoCo viewer. Use the standard GUI panels to inspect joints.")
    print("Press 'F1' inside the viewer to see all built-in shortcuts.")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            viewer.sync()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Path to an MJCF file")
    parser.add_argument(
        "--pose-file",
        type=Path,
        help="Optional path to a pose file (JSON, NPY, CSV) containing a single qpos vector.",
    )
    parser.add_argument(
        "--pose",
        type=float,
        nargs="*",
        help="Pose values to load directly from the command line (expects DOF floats).",
    )
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(str(args.model))
    dof = model.nq

    pose = None
    if args.pose_file is not None:
        pose = _load_pose(args.pose_file, dof)
    elif args.pose is not None:
        pose = _parse_pose(args.pose, dof)

    inspect_pose(model, pose)


if __name__ == "__main__":
    main()
