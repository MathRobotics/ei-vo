"""High-level helpers for common trajectory and rendering workflows."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .backends import KinematicsSpec, RenderSpec, coerce_kinematics_spec
from .core import Trajectory, load_angles
from .programs import generate_trajectory


def _merge_meta(
    base: Mapping[str, Any] | None,
    extra: Mapping[str, Any] | None,
) -> dict[str, Any]:
    merged = dict(base or {})
    if extra is not None:
        merged.update(extra)
    return merged


def trajectory_from_file(
    path: str | Path,
    *,
    deg: bool = False,
    hz: float | None = None,
    meta: Mapping[str, Any] | None = None,
) -> Trajectory:
    """Load a trajectory file into a validated :class:`Trajectory`."""

    positions = load_angles(path, deg=deg)
    dt = None if hz is None else 1.0 / hz
    return Trajectory.from_positions(positions, dt=dt, meta=dict(meta or {}))


def trajectory_from_program(
    dof: int,
    *,
    program: str = "waypoints",
    hz: float = 240.0,
    segment_duration: float = 1.5,
    duration: float = 10.0,
    meta: Mapping[str, Any] | None = None,
) -> Trajectory:
    """Generate a built-in trajectory program and attach optional metadata."""

    trajectory = generate_trajectory(
        dof,
        program=program,
        hz=hz,
        segment_duration=segment_duration,
        duration=duration,
    )
    trajectory.meta = _merge_meta(trajectory.meta, meta)
    return trajectory


def _load_render_model_dof(model_path: str | Path) -> int:
    from .render.render_mj import load_robot_model

    return load_robot_model(model_path).dof


def _load_kinematics_model_dof(
    spec: KinematicsSpec,
    *,
    model_path: str | Path | None = None,
) -> int:
    from .kinematics import load_model_dof

    resolved_model_path, kwargs = spec.resolve(model_path=model_path)
    return int(load_model_dof(spec.backend, resolved_model_path, **kwargs))


def resolve_program_dof(
    model_path: str | Path | None = None,
    *,
    dof: int | None = None,
    kinematics: str | KinematicsSpec | None = None,
) -> int:
    """Resolve the DOF needed to build a built-in trajectory program."""

    if model_path is not None:
        return _load_render_model_dof(model_path)

    if dof is not None:
        if dof <= 0:
            raise ValueError(f"dof must be positive. Got {dof}.")
        return dof

    resolved_kinematics = coerce_kinematics_spec(kinematics)
    if resolved_kinematics is not None:
        return _load_kinematics_model_dof(resolved_kinematics, model_path=model_path)

    raise ValueError("Specify either model_path, dof, or a kinematics spec with model_path.")


def _validate_trajectory_dof(
    trajectory: Trajectory,
    *,
    model_path: str | Path | None = None,
    kinematics: str | KinematicsSpec | None = None,
) -> None:
    expected_dof = None
    if model_path is not None:
        expected_dof = _load_render_model_dof(model_path)
    else:
        resolved_kinematics = coerce_kinematics_spec(kinematics)
        if resolved_kinematics is not None and resolved_kinematics.model_path is not None:
            expected_dof = _load_kinematics_model_dof(resolved_kinematics)

    if expected_dof is not None and trajectory.dof != expected_dof:
        raise ValueError(
            f"Trajectory DOF ({trajectory.dof}) does not match model DOF ({expected_dof})."
        )


def render_program(
    model_path: str | Path | None = None,
    *,
    dof: int | None = None,
    program: str = "waypoints",
    renderer: str | RenderSpec = "mujoco",
    hz: float = 240.0,
    slow: float = 1.0,
    camera=None,
    loop: bool = False,
    segment_duration: float = 1.5,
    duration: float = 10.0,
    record_path: str | Path | None = None,
    record_fps: float | None = None,
    record_size: tuple[int, int] | None = None,
    kinematics: str | KinematicsSpec | None = None,
    **backend_kwargs,
) -> Trajectory:
    """Generate a built-in program and render it with the selected backend."""

    resolved_dof = resolve_program_dof(model_path, dof=dof, kinematics=kinematics)
    trajectory = trajectory_from_program(
        resolved_dof,
        program=program,
        hz=hz,
        segment_duration=segment_duration,
        duration=duration,
    )

    from .render.play import play as render_trajectory

    render_trajectory(
        model_path,
        trajectory,
        slow=slow,
        hz=hz,
        camera=camera,
        loop=loop,
        record_path=record_path,
        record_fps=record_fps,
        record_size=record_size,
        renderer=renderer,
        kinematics=kinematics,
        **backend_kwargs,
    )
    return trajectory


def render_angles(
    path: str | Path,
    *,
    model_path: str | Path | None = None,
    deg: bool = False,
    renderer: str | RenderSpec = "mujoco",
    hz: float = 240.0,
    slow: float = 1.0,
    camera=None,
    loop: bool = False,
    record_path: str | Path | None = None,
    record_fps: float | None = None,
    record_size: tuple[int, int] | None = None,
    kinematics: str | KinematicsSpec | None = None,
    meta: Mapping[str, Any] | None = None,
    **backend_kwargs,
) -> Trajectory:
    """Load a trajectory file and render it with the selected backend."""

    trajectory = trajectory_from_file(path, deg=deg, hz=hz, meta=meta)
    _validate_trajectory_dof(trajectory, model_path=model_path, kinematics=kinematics)

    from .render.play import play as render_trajectory

    render_trajectory(
        model_path,
        trajectory,
        slow=slow,
        hz=hz,
        camera=camera,
        loop=loop,
        record_path=record_path,
        record_fps=record_fps,
        record_size=record_size,
        renderer=renderer,
        kinematics=kinematics,
        **backend_kwargs,
    )
    return trajectory


__all__ = [
    "render_angles",
    "render_program",
    "resolve_program_dof",
    "trajectory_from_file",
    "trajectory_from_program",
]
