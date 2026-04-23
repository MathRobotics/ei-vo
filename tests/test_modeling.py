import importlib
import pathlib

import numpy as np
import pytest


def _write_demo_urdf(path: pathlib.Path) -> pathlib.Path:
    path.write_text(
        """<?xml version="1.0"?>
<robot name="demo_arm">
  <material name="orange">
    <color rgba="0.9 0.4 0.1 1.0"/>
  </material>
  <link name="base">
    <visual>
      <origin xyz="0 0 0.05" rpy="0 0 0"/>
      <geometry><box size="0.2 0.2 0.1"/></geometry>
    </visual>
  </link>
  <link name="link1">
    <visual>
      <origin xyz="0 0 0.15" rpy="0 0 0"/>
      <geometry><cylinder radius="0.03" length="0.3"/></geometry>
      <material name="orange"/>
    </visual>
  </link>
  <link name="ee">
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><sphere radius="0.02"/></geometry>
    </visual>
  </link>
  <joint name="joint1" type="revolute">
    <parent link="base"/>
    <child link="link1"/>
    <origin xyz="0 0 0.1" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.5" upper="1.5" effort="1" velocity="1"/>
  </joint>
  <joint name="joint2" type="continuous">
    <parent link="link1"/>
    <child link="ee"/>
    <origin xyz="0 0 0.3" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
  </joint>
</robot>
""",
        encoding="utf-8",
    )
    return path


def test_load_robot_model_parses_urdf(tmp_path: pathlib.Path):
    modeling = importlib.import_module("ei_vo.modeling")
    urdf_path = _write_demo_urdf(tmp_path / "robot.urdf")

    robot = modeling.load_robot_model(urdf_path)

    assert robot.name == "demo_arm"
    assert robot.joint_names == ("joint1", "joint2")
    assert robot.dof == 2
    np.testing.assert_allclose(robot.limits[0], [-1.5, 1.5])
    assert np.isneginf(robot.limits[1, 0])
    assert np.isposinf(robot.limits[1, 1])


def test_load_urdf_scene_and_compute_link_poses(tmp_path: pathlib.Path):
    modeling = importlib.import_module("ei_vo.modeling")
    urdf_path = _write_demo_urdf(tmp_path / "robot.urdf")

    scene = modeling.load_urdf_scene(urdf_path)
    poses = modeling.compute_link_poses(scene, [0.5, 0.25])

    assert scene.root_link == "base"
    assert tuple(sorted(poses)) == ("base", "ee", "link1")
    np.testing.assert_allclose(poses["base"][:3, 3], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(poses["link1"][:3, 3], [0.0, 0.0, 0.1])
    np.testing.assert_allclose(poses["ee"][:3, 3], [0.0, 0.0, 0.4], atol=1e-6)


def test_load_robot_model_rejects_non_urdf(tmp_path: pathlib.Path):
    modeling = importlib.import_module("ei_vo.modeling")
    xml_path = tmp_path / "robot.xml"
    xml_path.write_text("<robot/>", encoding="utf-8")

    with pytest.raises(ValueError, match="Only URDF models are supported"):
        modeling.load_robot_model(xml_path)
