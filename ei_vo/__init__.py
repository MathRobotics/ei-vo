"""Public package interface with lazy exports."""

from __future__ import annotations

import importlib

_LAZY_EXPORTS = {
    "CameraSettings": (".config", "CameraSettings"),
    "KinematicsSpec": (".backends", "KinematicsSpec"),
    "PlaybackConfig": (".config", "PlaybackConfig"),
    "RecordingConfig": (".config", "RecordingConfig"),
    "RenderSpec": (".backends", "RenderSpec"),
    "available_programs": (".programs", "available_programs"),
    "build_demo_trajectory": (".demo", "build_demo_trajectory"),
    "build_sine_demo": (".demo", "build_sine_demo"),
    "build_sine_trajectory": (".programs", "build_sine_trajectory"),
    "build_waypoint_trajectory": (".programs", "build_waypoint_trajectory"),
    "default_waypoints": (".programs", "default_waypoints"),
    "demo_waypoints": (".demo", "demo_waypoints"),
    "generate_demo_trajectory": (".demo", "generate_demo_trajectory"),
    "generate_positions": (".programs", "generate_positions"),
    "generate_trajectory": (".programs", "generate_trajectory"),
    "get_kinematics_backend": (".kinematics.registry", "get_kinematics_backend"),
    "get_renderer": (".render.registry", "get_renderer"),
    "load_angles": (".core.angles", "load_angles"),
    "load_camera_settings": (".config", "load_camera_settings"),
    "RobotModel": (".core.core", "RobotModel"),
    "Trajectory": (".core.core", "Trajectory"),
    "available_kinematics_backends": (".kinematics.registry", "available_kinematics_backends"),
    "available_renderers": (".render.registry", "available_renderers"),
    "ModelSource": (".modeling", "ModelSource"),
    "load_robot_model": (".modeling", "load_robot_model"),
    "quintic": (".core.interpolation", "quintic"),
    "register_kinematics_backend": (".kinematics.registry", "register_kinematics_backend"),
    "register_renderer": (".render.registry", "register_renderer"),
    "render_angles": (".workflows", "render_angles"),
    "render_program": (".workflows", "render_program"),
    "resolve_program_dof": (".workflows", "resolve_program_dof"),
    "resolve_record_destination": (".core.recording", "resolve_record_destination"),
    "save_camera_settings": (".config", "save_camera_settings"),
    "trajectory_from_file": (".workflows", "trajectory_from_file"),
    "trajectory_from_program": (".workflows", "trajectory_from_program"),
}


def play(*args, **kwargs):
    from .render.play import play as _play

    return _play(*args, **kwargs)


def render_trajectory(*args, **kwargs):
    from .render.play import play as _play

    return _play(*args, **kwargs)


def play_trajectory(*args, **kwargs):
    from .render.play import play as _play_trajectory

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


def __getattr__(name: str):
    try:
        module_name, attr_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    value = getattr(importlib.import_module(module_name, __name__), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


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
    "load_camera_settings",
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
    "save_camera_settings",
    "trajectory_from_file",
    "trajectory_from_program",
]
