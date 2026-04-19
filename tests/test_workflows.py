import importlib
import pathlib
import sys

import numpy as np
import pytest

from ei_vo import KinematicsSpec, RenderSpec
from ei_vo.core import Trajectory


def test_example_bootstrap_inserts_repo_root(monkeypatch):
    bootstrap = importlib.import_module("examples._bootstrap")
    monkeypatch.setattr(sys, "path", ["/tmp/site-packages"], raising=False)

    repo_root = bootstrap.ensure_repo_root_on_path()

    assert sys.path[0] == repo_root.as_posix()


def test_trajectory_from_file_uses_hz_and_meta(tmp_path: pathlib.Path):
    workflows = importlib.import_module("ei_vo.workflows")
    csv_path = tmp_path / "angles.csv"
    np.savetxt(csv_path, np.array([[0.0, 1.0], [2.0, 3.0]], dtype=float), delimiter=",", fmt="%.6f")

    trajectory = workflows.trajectory_from_file(csv_path, hz=20.0, meta={"name": "sample"})

    assert isinstance(trajectory, Trajectory)
    assert trajectory.meta == {"name": "sample"}
    np.testing.assert_allclose(trajectory.t, [0.0, 0.05])


def test_trajectory_from_program_merges_meta():
    workflows = importlib.import_module("ei_vo.workflows")

    trajectory = workflows.trajectory_from_program(
        3,
        program="sine",
        hz=10.0,
        duration=1.0,
        meta={"name": "sample"},
    )

    assert trajectory.meta["program"] == "sine"
    assert trajectory.meta["name"] == "sample"


def test_resolve_program_dof_can_use_kinematics_spec(monkeypatch):
    workflows = importlib.import_module("ei_vo.workflows")
    monkeypatch.setattr(workflows, "_load_kinematics_model_dof", lambda spec, model_path=None: 6)

    dof = workflows.resolve_program_dof(
        kinematics=KinematicsSpec("pinocchio", model_path="robot.urdf", base_link="base", end_link="ee")
    )

    assert dof == 6


def test_render_program_infers_dof_from_model_and_calls_renderer(monkeypatch):
    workflows = importlib.import_module("ei_vo.workflows")
    render_play = importlib.import_module("ei_vo.render.play")
    calls = {}

    monkeypatch.setattr(workflows, "_load_render_model_dof", lambda path: 3)

    def fake_play(model_path, trajectory, **kwargs):
        calls["model_path"] = model_path
        calls["trajectory"] = trajectory
        calls["kwargs"] = kwargs

    monkeypatch.setattr(render_play, "play", fake_play)

    trajectory = workflows.render_program("robot.xml", program="waypoints", hz=20.0)

    assert isinstance(trajectory, Trajectory)
    assert calls["model_path"] == "robot.xml"
    assert calls["trajectory"].dof == 3
    assert calls["kwargs"]["renderer"] == "mujoco"


def test_render_play_resolves_kinematics_spec(monkeypatch):
    render_play = importlib.import_module("ei_vo.render.play")
    calls = {}

    def fake_dispatch(renderer, /, **kwargs):
        calls["renderer"] = renderer
        calls["kwargs"] = kwargs

    monkeypatch.setattr(render_play, "dispatch_render", fake_dispatch)

    render_play.play(
        "robot.xml",
        [[0.0, 1.0]],
        renderer="mujoco",
        kinematics=KinematicsSpec("pinocchio", model_path="robot.urdf", base_link="base", end_link="ee"),
    )

    assert calls["renderer"] == "mujoco"
    assert calls["kwargs"]["kinematics_backend"] == "pinocchio"
    assert calls["kwargs"]["kinematics_model_path"] == "robot.urdf"
    assert calls["kwargs"]["base_link"] == "base"
    assert calls["kwargs"]["end_link"] == "ee"


def test_render_angles_validates_model_dof(monkeypatch, tmp_path: pathlib.Path):
    workflows = importlib.import_module("ei_vo.workflows")
    csv_path = tmp_path / "angles.csv"
    np.savetxt(csv_path, np.array([[0.0, 1.0]], dtype=float), delimiter=",", fmt="%.6f")
    monkeypatch.setattr(workflows, "_load_render_model_dof", lambda path: 3)

    with pytest.raises(ValueError, match="Trajectory DOF"):
        workflows.render_angles(csv_path, model_path="robot.xml")


def test_switch_renderer_example_supports_matplotlib(monkeypatch):
    module = importlib.import_module("examples.switch_renderer")
    calls = {}

    def fake_render_program(*args, **kwargs):
        calls["args"] = args
        calls["kwargs"] = kwargs

    monkeypatch.setattr(module, "render_program", fake_render_program)
    monkeypatch.setattr(module, "RENDERER", "matplotlib")
    module.main()

    assert str(calls["args"][0]).endswith("examples/models/three_dof_arm.urdf")
    assert isinstance(calls["kwargs"]["renderer"], RenderSpec)
    assert calls["kwargs"]["renderer"].renderer == "matplotlib"
    assert isinstance(calls["kwargs"]["kinematics"], KinematicsSpec)
    assert calls["kwargs"]["kinematics"].backend == "pinocchio"


def test_switch_renderer_example_defaults_to_meshcat_and_pinocchio(monkeypatch):
    module = importlib.import_module("examples.switch_renderer")
    calls = {}

    def fake_render_program(*args, **kwargs):
        calls["args"] = args
        calls["kwargs"] = kwargs

    monkeypatch.setattr(module, "render_program", fake_render_program)
    module.main()

    assert str(calls["args"][0]).endswith("examples/models/three_dof_arm.urdf")
    assert calls["kwargs"]["renderer"] == "meshcat"
    assert isinstance(calls["kwargs"]["kinematics"], KinematicsSpec)
    assert calls["kwargs"]["kinematics"].backend == "pinocchio"
    assert calls["kwargs"]["kinematics"].base_link == "base"
    assert calls["kwargs"]["kinematics"].end_link == "ee"


def test_switch_renderer_example_supports_kinematics_backend(monkeypatch):
    module = importlib.import_module("examples.switch_renderer")
    calls = {}

    def fake_render_program(*args, **kwargs):
        calls["args"] = args
        calls["kwargs"] = kwargs

    monkeypatch.setattr(module, "render_program", fake_render_program)
    monkeypatch.setattr(
        module,
        "maybe_relaunch_with_mjpython",
        lambda renderer, *, exec_args: calls.setdefault("mjpython", (renderer, exec_args)),
    )
    monkeypatch.setattr(module, "RENDERER", "mujoco")
    monkeypatch.setattr(module, "BACKEND", "pinocchio")
    monkeypatch.setattr(module, "MODEL", pathlib.Path("robot.urdf"))
    monkeypatch.setattr(module, "BASE_LINK", "base")
    monkeypatch.setattr(module, "END_LINK", "ee")
    module.main()

    assert calls["args"][0] == pathlib.Path("robot.urdf")
    assert calls["kwargs"]["renderer"] == "mujoco"
    assert isinstance(calls["kwargs"]["kinematics"], KinematicsSpec)
    assert calls["kwargs"]["kinematics"].backend == "pinocchio"
    assert calls["kwargs"]["kinematics"].model_path is None
    assert calls["kwargs"]["kinematics"].base_link == "base"
    assert calls["kwargs"]["kinematics"].end_link == "ee"
    assert calls["mjpython"][0] == "mujoco"
    assert calls["mjpython"][1][0].endswith("examples/switch_renderer.py")


def test_switch_renderer_example_supports_blender(monkeypatch):
    module = importlib.import_module("examples.switch_renderer")
    calls = {}

    def fake_render_program(*args, **kwargs):
        calls["args"] = args
        calls["kwargs"] = kwargs

    monkeypatch.setattr(module, "render_program", fake_render_program)
    monkeypatch.setattr(module, "RENDERER", "blender")
    module.main()

    assert str(calls["args"][0]).endswith("examples/models/three_dof_arm.urdf")
    assert calls["kwargs"]["renderer"] == "blender"
    assert str(calls["kwargs"]["record_path"]).endswith("recordings/switch_renderer_blender.mp4")
