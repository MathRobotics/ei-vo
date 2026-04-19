import importlib
import pathlib

import numpy as np


def test_load_robot_model_parses_urdf_without_mujoco(monkeypatch, tmp_path: pathlib.Path):
    modeling = importlib.import_module("ei_vo.modeling")
    mujoco_modeling = importlib.import_module("ei_vo.modeling.mujoco")

    urdf_path = tmp_path / "robot.urdf"
    urdf_path.write_text(
        """<?xml version="1.0"?>
<robot name="demo_arm">
  <link name="base"/>
  <link name="link1"/>
  <link name="link2"/>
  <joint name="joint2" type="continuous">
    <parent link="link1"/>
    <child link="link2"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
  </joint>
  <joint name="joint1" type="revolute">
    <parent link="base"/>
    <child link="link1"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.5" upper="1.5" effort="1" velocity="1"/>
  </joint>
</robot>
""",
        encoding="utf-8",
    )

    def fail_import():
        raise AssertionError("URDF metadata loading should not require MuJoCo.")

    monkeypatch.setattr(mujoco_modeling, "_import_mujoco", fail_import)

    robot = modeling.load_robot_model(urdf_path)

    assert robot.name == "demo_arm"
    assert robot.joint_names == ("joint1", "joint2")
    assert robot.dof == 2
    np.testing.assert_allclose(robot.limits[0], [-1.5, 1.5])
    assert np.isneginf(robot.limits[1, 0])
    assert np.isposinf(robot.limits[1, 1])
