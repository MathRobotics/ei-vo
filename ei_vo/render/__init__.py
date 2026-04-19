"""Renderer package exports with lazy backend loading."""

from __future__ import annotations

from ..config import CameraSettings, PlaybackConfig, RecordingConfig
from .play import play
from .registry import available_renderers, get_renderer, register_renderer

__all__ = [
    "CameraSettings",
    "PlaybackConfig",
    "RecordingConfig",
    "available_renderers",
    "get_renderer",
    "play",
    "register_renderer",
]


def __getattr__(name: str):
    if name in {
        "ArmJointMap",
        "clamp_to_limits",
        "detect_arm_joint_qaddr",
        "detect_arm_joints",
        "load_robot_model",
        "play_trajectory",
    }:
        from .render_mj import (
            ArmJointMap,
            clamp_to_limits,
            detect_arm_joint_qaddr,
            detect_arm_joints,
            load_robot_model,
            play_trajectory,
        )

        return {
            "ArmJointMap": ArmJointMap,
            "clamp_to_limits": clamp_to_limits,
            "detect_arm_joint_qaddr": detect_arm_joint_qaddr,
            "detect_arm_joints": detect_arm_joints,
            "load_robot_model": load_robot_model,
            "play_trajectory": play_trajectory,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
