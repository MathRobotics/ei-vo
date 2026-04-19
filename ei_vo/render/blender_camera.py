"""Pure-Python camera helpers shared by the Blender renderer and tests."""

from __future__ import annotations

import math

_DEFAULT_CAMERA_LENS_MM = 28.0
_DEFAULT_SENSOR_WIDTH_MM = 36.0
_DEFAULT_SENSOR_HEIGHT_MM = 24.0
_DEFAULT_CAMERA_MARGIN = 1.15
_DEFAULT_CAMERA_MIN_DISTANCE = 0.75


def default_blender_camera_distance(
    radius: float,
    *,
    width: int,
    height: int,
    lens_mm: float = _DEFAULT_CAMERA_LENS_MM,
    sensor_width_mm: float = _DEFAULT_SENSOR_WIDTH_MM,
    sensor_height_mm: float = _DEFAULT_SENSOR_HEIGHT_MM,
    margin: float = _DEFAULT_CAMERA_MARGIN,
    min_distance: float = _DEFAULT_CAMERA_MIN_DISTANCE,
) -> float:
    """Return a conservative camera distance that fits the scene bounds."""

    scene_radius = max(float(radius), 0.25)
    aspect_ratio = max(float(width) / max(float(height), 1.0), 1e-6)

    effective_width = min(float(sensor_width_mm), float(sensor_height_mm) * aspect_ratio)
    effective_height = min(float(sensor_height_mm), float(sensor_width_mm) / aspect_ratio)
    half_fov = min(
        math.atan(effective_width / (2.0 * float(lens_mm))),
        math.atan(effective_height / (2.0 * float(lens_mm))),
    )
    if half_fov <= 0.0:
        return max(scene_radius * 3.0, float(min_distance))

    fitted_distance = scene_radius * float(margin) / math.tan(half_fov)
    return max(fitted_distance, float(min_distance))


__all__ = ["default_blender_camera_distance"]
