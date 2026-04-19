import pytest

from ei_vo.render.blender_camera import default_blender_camera_distance


def test_default_blender_camera_distance_zooms_small_scenes_closer():
    distance = default_blender_camera_distance(0.27, width=1280, height=720)

    assert distance == pytest.approx(0.8586666666666667, rel=1e-6)
    assert distance < 1.0


def test_default_blender_camera_distance_scales_with_scene_radius():
    small = default_blender_camera_distance(0.27, width=1280, height=720)
    large = default_blender_camera_distance(1.0, width=1280, height=720)

    assert large > small
    assert large == pytest.approx(3.180246913580247, rel=1e-6)


def test_default_blender_camera_distance_respects_minimum_distance():
    distance = default_blender_camera_distance(0.01, width=1280, height=720)

    assert distance == pytest.approx(0.7950617283950617, rel=1e-6)
