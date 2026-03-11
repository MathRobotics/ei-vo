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


def _install_dummy_pinocchio(monkeypatch):
    class DummyPlacement:
        def __init__(self, translation):
            self.translation = np.asarray(translation, dtype=float)
            self.rotation = np.eye(3, dtype=float)

    class DummyModel:
        nq = 2

        def __init__(self):
            self.frame_ids = {"base": 0, "ee": 1}

        def createData(self):
            return types.SimpleNamespace(
                oMf=[DummyPlacement([0.0, 0.0, 0.0]), DummyPlacement([0.0, 0.0, 0.0])]
            )

        def existFrame(self, name):
            return name in self.frame_ids

        def getFrameId(self, name):
            return self.frame_ids[name]

    def frames_forward_kinematics(model, data, q):
        data.oMf[0] = DummyPlacement([0.0, 0.0, 0.0])
        data.oMf[1] = DummyPlacement([q[0], q[1], q[0] - q[1]])

    module = types.ModuleType("pinocchio")
    module.buildModelFromUrdf = lambda path: DummyModel()
    module.buildModelFromMJCF = lambda path: DummyModel()
    module.framesForwardKinematics = frames_forward_kinematics
    monkeypatch.setitem(sys.modules, "pinocchio", module)


def test_available_kinematics_backends():
    sys.modules.pop("ei_vo.kinematics", None)
    sys.modules.pop("ei_vo.kinematics.registry", None)
    kinematics = importlib.import_module("ei_vo.kinematics")

    assert kinematics.available_kinematics_backends() == ("literobo", "pinocchio")


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


def test_pinocchio_forward_kinematics(monkeypatch, tmp_path):
    _install_dummy_pinocchio(monkeypatch)
    sys.modules.pop("ei_vo.kinematics.pinocchio_backend", None)
    sys.modules.pop("ei_vo.kinematics.registry", None)
    kinematics = importlib.import_module("ei_vo.kinematics")

    model_path = tmp_path / "robot.urdf"
    model_path.write_text("<robot/>", encoding="utf-8")
    result = kinematics.forward_kinematics(
        "pinocchio",
        model_path,
        [[1.0, 2.0], [3.0, 4.0]],
        base_link="base",
        end_link="ee",
    )

    assert result.backend == "pinocchio"
    np.testing.assert_allclose(result.positions, [[1.0, 2.0, -1.0], [3.0, 4.0, -1.0]])
    assert kinematics.load_model_dof("pinocchio", model_path) == 2


def test_top_level_forward_kinematics_accepts_spec(monkeypatch, tmp_path):
    _install_dummy_pinocchio(monkeypatch)
    sys.modules.pop("ei_vo.kinematics.pinocchio_backend", None)
    sys.modules.pop("ei_vo.kinematics.registry", None)

    model_path = tmp_path / "robot.urdf"
    model_path.write_text("<robot/>", encoding="utf-8")
    result = ei_vo.forward_kinematics(
        KinematicsSpec("pinocchio", model_path=model_path, base_link="base", end_link="ee"),
        [[1.0, 2.0], [3.0, 4.0]],
    )

    assert result.backend == "pinocchio"
    np.testing.assert_allclose(result.positions, [[1.0, 2.0, -1.0], [3.0, 4.0, -1.0]])
