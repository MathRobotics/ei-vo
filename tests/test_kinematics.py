import importlib
import sys
import types

import numpy as np

import ei_vo
from ei_vo import KinematicsSpec
from ei_vo.core import Trajectory


def _install_dummy_literobo(monkeypatch):
    class DummyRobot:
        def __init__(self, base_link, end_link):
            self.base_link = base_link
            self.end_link = end_link
            self.dof = 2

        def forward_kinematics(self, joints):
            transform = np.eye(4, dtype=float)
            transform[:3, 3] = [joints[0], joints[1], joints[0] + joints[1]]
            return transform

    module = types.ModuleType("literobo")
    module.from_urdf_file = lambda path, base_link, end_link: DummyRobot(base_link, end_link)
    monkeypatch.setitem(sys.modules, "literobo", module)

def test_available_kinematics_backends():
    sys.modules.pop("ei_vo.kinematics", None)
    sys.modules.pop("ei_vo.kinematics.registry", None)
    kinematics = importlib.import_module("ei_vo.kinematics")

    assert kinematics.available_kinematics_backends() == ("literobo",)


def test_literobo_forward_kinematics(monkeypatch, tmp_path):
    _install_dummy_literobo(monkeypatch)
    sys.modules.pop("ei_vo.kinematics.literobo_backend", None)
    sys.modules.pop("ei_vo.kinematics.registry", None)
    kinematics = importlib.import_module("ei_vo.kinematics")

    model_path = tmp_path / "robot.urdf"
    model_path.write_text("<robot/>", encoding="utf-8")
    trajectory = Trajectory.from_positions([[0.1, 0.2], [0.3, 0.4]])
    result = kinematics.forward_kinematics(
        "literobo",
        model_path,
        trajectory,
        base_link="base",
        end_link="ee",
    )

    assert result.backend == "literobo"
    np.testing.assert_allclose(result.positions, [[0.1, 0.2, 0.3], [0.3, 0.4, 0.7]])
    assert kinematics.load_model_dof(
        "literobo",
        model_path,
        base_link="base",
        end_link="ee",
    ) == 2


def test_top_level_forward_kinematics_accepts_spec(monkeypatch, tmp_path):
    _install_dummy_literobo(monkeypatch)
    sys.modules.pop("ei_vo.kinematics.literobo_backend", None)
    sys.modules.pop("ei_vo.kinematics.registry", None)

    model_path = tmp_path / "robot.urdf"
    model_path.write_text("<robot/>", encoding="utf-8")
    result = ei_vo.forward_kinematics(
        KinematicsSpec("literobo", model_path=model_path, base_link="base", end_link="ee"),
        [[1.0, 2.0], [3.0, 4.0]],
    )

    assert result.backend == "literobo"
    np.testing.assert_allclose(result.positions, [[1.0, 2.0, 3.0], [3.0, 4.0, 7.0]])
