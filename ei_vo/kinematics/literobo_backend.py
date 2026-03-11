"""LiteRobo-based forward kinematics."""

from __future__ import annotations

import pathlib

import numpy as np

from .common import KinematicsResult, coerce_trajectory


def _require_chain_links(base_link: str | None, end_link: str | None) -> tuple[str, str]:
    if not base_link or not end_link:
        raise ValueError("LiteRobo kinematics requires both base_link and end_link.")
    return base_link, end_link


def _load_robot(model_path: str | pathlib.Path, *, base_link: str | None, end_link: str | None):
    try:
        import literobo
    except ImportError as exc:
        raise RuntimeError(
            "The 'literobo' kinematics backend requires the optional dependency 'literobo'."
        ) from exc

    base_name, end_name = _require_chain_links(base_link, end_link)
    path = pathlib.Path(model_path)
    if path.suffix.lower() != ".urdf":
        raise ValueError("LiteRobo currently supports URDF models only.")
    robot = literobo.from_urdf_file(path.as_posix(), base_name, end_name)
    return robot, path, base_name, end_name


def load_model_dof(
    model_path: str | pathlib.Path,
    *,
    base_link: str | None = None,
    end_link: str | None = None,
) -> int:
    """Return chain DOF for a LiteRobo model."""

    robot, _, _, _ = _load_robot(model_path, base_link=base_link, end_link=end_link)
    return int(robot.dof)


def forward_kinematics(
    model_path: str | pathlib.Path,
    traj,
    *,
    base_link: str | None = None,
    end_link: str | None = None,
) -> KinematicsResult:
    """Compute end-effector transforms for a trajectory with LiteRobo."""

    robot, path, base_name, end_name = _load_robot(
        model_path,
        base_link=base_link,
        end_link=end_link,
    )
    trajectory = coerce_trajectory(traj)
    if int(robot.dof) != trajectory.dof:
        raise ValueError(
            f"Trajectory dof ({trajectory.dof}) does not match LiteRobo chain dof ({robot.dof})."
        )

    transforms = np.zeros((trajectory.steps, 4, 4), dtype=float)
    for index, row in enumerate(trajectory.q):
        transforms[index] = np.asarray(robot.forward_kinematics(row), dtype=float)

    return KinematicsResult(
        transforms=transforms,
        backend="literobo",
        model_path=path.as_posix(),
        base_link=base_name,
        end_link=end_name,
    )


__all__ = ["forward_kinematics", "load_model_dof"]
