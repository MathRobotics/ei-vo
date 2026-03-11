"""Compatibility wrapper for the legacy demo naming."""

from __future__ import annotations

from typing import Literal

from .programs import (
    available_programs,
    build_sine_trajectory,
    build_waypoint_trajectory,
    default_waypoints,
    generate_positions,
    generate_trajectory,
    normalize_program_mode,
)

DemoMode = Literal["wp", "waypoints", "sine"]


def demo_waypoints(dof: int):
    """Backward-compatible alias for :func:`ei_vo.programs.default_waypoints`."""

    return default_waypoints(dof)


def build_demo_trajectory(waypoints, *, segment_duration: float, hz: float):
    """Backward-compatible alias for :func:`ei_vo.programs.build_waypoint_trajectory`."""

    return build_waypoint_trajectory(waypoints, segment_duration=segment_duration, hz=hz)


def build_sine_demo(dof: int, *, duration: float, hz: float):
    """Backward-compatible alias for :func:`ei_vo.programs.build_sine_trajectory`."""

    return build_sine_trajectory(dof, duration=duration, hz=hz)


def generate_demo_positions(
    dof: int,
    *,
    mode: DemoMode = "wp",
    hz: float = 240.0,
    segment_duration: float = 1.5,
    duration: float = 10.0,
):
    """Backward-compatible alias for :func:`ei_vo.programs.generate_positions`."""

    return generate_positions(
        dof,
        program=normalize_program_mode(mode),
        hz=hz,
        segment_duration=segment_duration,
        duration=duration,
    )


def generate_demo_trajectory(
    dof: int,
    *,
    mode: DemoMode = "wp",
    hz: float = 240.0,
    segment_duration: float = 1.5,
    duration: float = 10.0,
):
    """Backward-compatible alias for :func:`ei_vo.programs.generate_trajectory`."""

    trajectory = generate_trajectory(
        dof,
        program=normalize_program_mode(mode),
        hz=hz,
        segment_duration=segment_duration,
        duration=duration,
    )
    trajectory.meta.setdefault("demo_mode", mode)
    return trajectory


__all__ = [
    "DemoMode",
    "available_programs",
    "build_demo_trajectory",
    "build_sine_demo",
    "demo_waypoints",
    "generate_demo_positions",
    "generate_demo_trajectory",
]
