"""Built-in trajectory programs exposed as standard library features."""

from __future__ import annotations

import math
from typing import Literal

import numpy as np

from .core import Trajectory, quintic

ProgramMode = Literal["waypoints", "sine"]

_PROGRAM_ALIASES = {
    "wp": "waypoints",
    "waypoints": "waypoints",
    "sine": "sine",
}


def _validate_positive(name: str, value: float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive. Got {value}.")


def normalize_program_mode(mode: str) -> ProgramMode:
    """Normalize CLI and compatibility aliases into the canonical program names."""

    normalized = _PROGRAM_ALIASES.get(mode.strip().lower())
    if normalized is None:
        raise ValueError(
            f"Unsupported program mode: {mode!r}. Available programs: {', '.join(available_programs())}"
        )
    return normalized


def available_programs() -> tuple[ProgramMode, ...]:
    """List the built-in trajectory programs."""

    return ("sine", "waypoints")


def default_waypoints(dof: int) -> np.ndarray:
    """Return conservative default poses for ``dof`` joints."""

    if dof <= 0:
        raise ValueError(f"dof must be positive. Got {dof}.")

    base = np.linspace(-0.6, 0.6, dof, dtype=float)
    phase = np.linspace(0.0, math.pi, dof, dtype=float)

    offsets = [
        np.zeros(dof, dtype=float),
        0.35 * np.sin(phase),
        -0.25 * np.sin(phase + math.pi / 4.0),
        0.30 * np.sin(phase + math.pi / 2.0),
        np.zeros(dof, dtype=float),
    ]

    poses = [base + offset for offset in offsets]
    poses[-1] = poses[0].copy()
    return np.vstack(poses)


def build_waypoint_trajectory(waypoints: np.ndarray, *, segment_duration: float, hz: float) -> np.ndarray:
    """Connect waypoint pairs with quintic curves and concatenate the segments."""

    _validate_positive("segment_duration", segment_duration)
    _validate_positive("hz", hz)

    q_wp = np.asarray(waypoints, dtype=float)
    if q_wp.ndim != 2:
        raise ValueError(f"waypoints must be a 2D array. Got {q_wp.shape}.")
    if q_wp.shape[0] == 0:
        raise ValueError("waypoints must contain at least one row.")
    if q_wp.shape[0] == 1:
        return q_wp.copy()

    dt = 1.0 / hz
    chunks = []
    for index in range(q_wp.shape[0] - 1):
        segment = quintic(q_wp[index], q_wp[index + 1], segment_duration, dt)
        chunks.append(segment[:-1])
    chunks.append(q_wp[-1][None, :])
    return np.vstack(chunks)


def build_sine_trajectory(dof: int, *, duration: float, hz: float) -> np.ndarray:
    """Generate a simple sinusoidal trajectory within a conservative range."""

    if dof <= 0:
        raise ValueError(f"dof must be positive. Got {dof}.")
    _validate_positive("duration", duration)
    _validate_positive("hz", hz)

    dt = 1.0 / hz
    t = np.arange(0.0, duration + 1e-12, dt)
    q = np.zeros((t.shape[0], dof), dtype=float)
    base = np.linspace(-0.6, 0.6, dof, dtype=float)
    amp = np.linspace(0.15, 0.30, dof, dtype=float)
    freq = np.linspace(0.20, 0.35, dof, dtype=float)
    phase = np.linspace(0.0, math.pi, dof, dtype=float)
    for joint_index in range(dof):
        q[:, joint_index] = base[joint_index] + amp[joint_index] * np.sin(
            2.0 * math.pi * freq[joint_index] * t + phase[joint_index]
        )
    return q


def generate_positions(
    dof: int,
    *,
    program: str = "waypoints",
    hz: float = 240.0,
    segment_duration: float = 1.5,
    duration: float = 10.0,
) -> np.ndarray:
    """Generate playback positions for one of the built-in programs."""

    resolved_program = normalize_program_mode(program)
    if resolved_program == "waypoints":
        return build_waypoint_trajectory(
            default_waypoints(dof),
            segment_duration=segment_duration,
            hz=hz,
        )
    return build_sine_trajectory(dof, duration=duration, hz=hz)


def generate_trajectory(
    dof: int,
    *,
    program: str = "waypoints",
    hz: float = 240.0,
    segment_duration: float = 1.5,
    duration: float = 10.0,
) -> Trajectory:
    """Generate a built-in program as a validated :class:`Trajectory`."""

    resolved_program = normalize_program_mode(program)
    positions = generate_positions(
        dof,
        program=resolved_program,
        hz=hz,
        segment_duration=segment_duration,
        duration=duration,
    )
    return Trajectory.from_positions(
        positions,
        dt=1.0 / hz,
        meta={"program": resolved_program},
    )


__all__ = [
    "ProgramMode",
    "available_programs",
    "build_sine_trajectory",
    "build_waypoint_trajectory",
    "default_waypoints",
    "generate_positions",
    "generate_trajectory",
    "normalize_program_mode",
]
