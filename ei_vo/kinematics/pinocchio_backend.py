"""Pinocchio-based forward kinematics."""

from __future__ import annotations

import pathlib

import numpy as np

from .common import KinematicsResult, coerce_trajectory, make_transform


def _import_pinocchio():
    try:
        import pinocchio as pin
    except ImportError as exc:
        raise RuntimeError(
            "The 'pinocchio' kinematics backend requires the optional dependency 'pin'."
        ) from exc
    return pin


def _build_model(pin, model_path: str | pathlib.Path):
    path = pathlib.Path(model_path)
    suffix = path.suffix.lower()
    if suffix == ".urdf":
        return pin.buildModelFromUrdf(path.as_posix()), path
    raise ValueError(f"Unsupported model format for Pinocchio: {path.suffix}. Only .urdf is supported.")


def _placement_to_matrix(placement) -> np.ndarray:
    homogeneous = getattr(placement, "homogeneous", None)
    if homogeneous is not None:
        return np.asarray(homogeneous, dtype=float)
    return make_transform(
        rotation=np.asarray(placement.rotation, dtype=float),
        translation=np.asarray(placement.translation, dtype=float),
    )


def _resolve_frame_transform(model, data, name: str | None) -> np.ndarray:
    if name in {None, "", "world", "universe"}:
        return np.eye(4, dtype=float)

    if hasattr(model, "existFrame") and model.existFrame(name):
        return _placement_to_matrix(data.oMf[model.getFrameId(name)])

    if hasattr(model, "getFrameId"):
        try:
            frame_id = model.getFrameId(name)
        except Exception:
            frame_id = None
        if frame_id is not None and frame_id < len(getattr(data, "oMf", [])):
            return _placement_to_matrix(data.oMf[frame_id])

    if hasattr(model, "getJointId"):
        try:
            joint_id = model.getJointId(name)
        except Exception:
            joint_id = None
        if joint_id is not None and joint_id < len(getattr(data, "oMi", [])):
            return _placement_to_matrix(data.oMi[joint_id])

    raise ValueError(f"Could not resolve frame or joint {name!r} in the Pinocchio model.")


def _update_kinematics(pin, model, data, q: np.ndarray) -> None:
    if hasattr(pin, "framesForwardKinematics"):
        pin.framesForwardKinematics(model, data, q)
        return

    pin.forwardKinematics(model, data, q)
    if hasattr(pin, "updateFramePlacements"):
        pin.updateFramePlacements(model, data)


def load_model_dof(
    model_path: str | pathlib.Path,
    *,
    base_link: str | None = None,
    end_link: str | None = None,
) -> int:
    """Return model DOF for Pinocchio."""

    del base_link, end_link

    pin = _import_pinocchio()
    model, _ = _build_model(pin, model_path)
    return int(getattr(model, "nq"))


def forward_kinematics(
    model_path: str | pathlib.Path,
    traj,
    *,
    base_link: str | None = None,
    end_link: str | None = None,
) -> KinematicsResult:
    """Compute relative transforms between base and end link with Pinocchio."""

    if not end_link:
        raise ValueError("Pinocchio kinematics requires end_link.")

    pin = _import_pinocchio()
    model, path = _build_model(pin, model_path)
    trajectory = coerce_trajectory(traj)
    model_dof = int(getattr(model, "nq"))
    if model_dof != trajectory.dof:
        raise ValueError(
            f"Trajectory dof ({trajectory.dof}) does not match Pinocchio model dof ({model_dof})."
        )

    data = model.createData()
    transforms = np.zeros((trajectory.steps, 4, 4), dtype=float)
    for index, row in enumerate(trajectory.q):
        _update_kinematics(pin, model, data, np.asarray(row, dtype=float))
        base_transform = _resolve_frame_transform(model, data, base_link)
        end_transform = _resolve_frame_transform(model, data, end_link)
        transforms[index] = np.linalg.inv(base_transform) @ end_transform

    return KinematicsResult(
        transforms=transforms,
        backend="pinocchio",
        model_path=path.as_posix(),
        base_link=base_link,
        end_link=end_link,
    )


__all__ = ["forward_kinematics", "load_model_dof"]
