"""Compatibility wrapper for the canonical MuJoCo renderer module."""

from .render_mj import (
    ArmJointMap,
    CameraSettings,
    PlaybackConfig,
    RecordingConfig,
    clamp_to_limits,
    detect_arm_joint_qaddr,
    detect_arm_joints,
    load_robot_model,
    play,
    play_trajectory,
)

__all__ = [
    "ArmJointMap",
    "CameraSettings",
    "PlaybackConfig",
    "RecordingConfig",
    "clamp_to_limits",
    "detect_arm_joint_qaddr",
    "detect_arm_joints",
    "load_robot_model",
    "play",
    "play_trajectory",
]
