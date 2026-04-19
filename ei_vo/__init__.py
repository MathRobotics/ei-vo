"""Public package interface."""

from __future__ import annotations

from .backends import KinematicsSpec, RenderSpec
from .config import CameraSettings, PlaybackConfig, RecordingConfig
from .core.angles import load_angles
from .core.core import RobotModel, Trajectory
from .core.interpolation import quintic
from .core.recording import resolve_record_destination
from .demo import build_demo_trajectory, build_sine_demo, demo_waypoints, generate_demo_trajectory
from .kinematics.registry import (
    available_kinematics_backends,
    get_kinematics_backend,
    register_kinematics_backend,
)
from .modeling import ModelSource, load_robot_model
from .programs import (
    available_programs,
    build_sine_trajectory,
    build_waypoint_trajectory,
    default_waypoints,
    generate_positions,
    generate_trajectory,
)
from .render.registry import available_renderers, get_renderer, register_renderer
from .workflows import (
    render_angles,
    render_program,
    resolve_program_dof,
    trajectory_from_file,
    trajectory_from_program,
)


def play(*args, **kwargs):
    from .render.play import play as _play

    return _play(*args, **kwargs)


def render_trajectory(*args, **kwargs):
    from .render.play import play as _play

    return _play(*args, **kwargs)


def play_trajectory(*args, **kwargs):
    from .render.render_mj import play_trajectory as _play_trajectory

    return _play_trajectory(*args, **kwargs)


def forward_kinematics(backend, *args, **kwargs):
    from .backends import KinematicsSpec
    from .kinematics import forward_kinematics as _forward_kinematics

    if isinstance(backend, KinematicsSpec):
        if len(args) == 1:
            model_path = None
            traj = args[0]
        elif len(args) == 2:
            model_path, traj = args
        else:
            raise TypeError(
                "forward_kinematics(KinematicsSpec, ...) expects (traj) or (model_path, traj)."
            )
        resolved_model_path, resolved_kwargs = backend.resolve(model_path=model_path)
        resolved_kwargs.update(kwargs)
        return _forward_kinematics(backend.backend, resolved_model_path, traj, **resolved_kwargs)
    return _forward_kinematics(backend, *args, **kwargs)
__all__ = [
    "CameraSettings",
    "KinematicsSpec",
    "PlaybackConfig",
    "RecordingConfig",
    "RenderSpec",
    "available_programs",
    "build_sine_trajectory",
    "build_waypoint_trajectory",
    "default_waypoints",
    "forward_kinematics",
    "generate_positions",
    "generate_trajectory",
    "get_kinematics_backend",
    "get_renderer",
    "load_angles",
    "RobotModel",
    "Trajectory",
    "available_kinematics_backends",
    "available_renderers",
    "ModelSource",
    "load_robot_model",
    "play",
    "play_trajectory",
    "quintic",
    "register_kinematics_backend",
    "register_renderer",
    "render_angles",
    "render_program",
    "render_trajectory",
    "resolve_program_dof",
    "resolve_record_destination",
    "trajectory_from_file",
    "trajectory_from_program",
]
