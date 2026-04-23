import importlib
import json
import math
import pathlib

import numpy as np
import pytest

import ei_vo
from ei_vo import KinematicsSpec, build_sine_trajectory, build_waypoint_trajectory, default_waypoints, generate_trajectory
from ei_vo.core import RobotModel, Trajectory, load_angles, quintic, resolve_record_destination
from ei_vo.demo import (
    build_demo_trajectory,
    build_sine_demo,
    demo_waypoints,
    generate_demo_trajectory,
)
from ei_vo.programs import available_programs, generate_positions

_EXAMPLE_MODEL = pathlib.Path(__file__).resolve().parents[1] / "examples/models/three_dof_arm.urdf"


def test_load_angles_csv_in_degrees_single_row(tmp_path: pathlib.Path):
    data = np.array([[0.0, 90.0, 180.0]], dtype=float)
    csv_path = tmp_path / "angles.csv"
    np.savetxt(csv_path, data, delimiter=",", fmt="%.6f")

    loaded = load_angles(csv_path, deg=True)

    assert loaded.shape == (1, 3)
    np.testing.assert_allclose(loaded, np.deg2rad(data))


def test_quintic_interpolation_hits_endpoints():
    q0 = np.zeros(4)
    q1 = np.ones(4) * math.pi

    segment = quintic(q0, q1, T=1.0, dt=0.2)

    assert segment.shape == (6, 4)
    np.testing.assert_allclose(segment[0], q0)
    np.testing.assert_allclose(segment[-1], q1)
    assert np.all(segment[1:] >= segment[:-1] - 1e-9)


def test_trajectory_from_positions_tracks_shape_and_time():
    trajectory = Trajectory.from_positions([0.1, 0.2, 0.3], dt=0.5, meta={"name": "sample"})

    assert trajectory.steps == 1
    assert trajectory.dof == 3
    np.testing.assert_allclose(trajectory.q, [[0.1, 0.2, 0.3]])
    np.testing.assert_allclose(trajectory.t, [0.0])
    assert trajectory.meta == {"name": "sample"}


def test_trajectory_validates_derivative_shapes():
    with pytest.raises(ValueError):
        Trajectory(q=np.zeros((2, 3)), dq=np.zeros((2, 2)))


def test_robot_model_clamp_respects_limits():
    robot = RobotModel(
        name="arm",
        joint_names=("j1", "j2"),
        limits=np.array([[-1.0, 1.0], [-0.5, 0.5]]),
    )

    clipped = robot.clamp(np.array([[2.0, -1.0], [-2.0, 1.0]]))

    np.testing.assert_allclose(clipped, [[1.0, -0.5], [-1.0, 0.5]])


def test_build_demo_trajectory_concatenates_segments():
    waypoints = np.array([
        np.zeros(3),
        np.ones(3),
        np.ones(3) * 2.0,
    ])

    trajectory = build_demo_trajectory(waypoints, segment_duration=1.0, hz=2.0)

    assert trajectory.shape == (5, 3)
    np.testing.assert_allclose(trajectory[0], waypoints[0])
    np.testing.assert_allclose(trajectory[-1], waypoints[-1])


def test_build_waypoint_trajectory_concatenates_segments():
    waypoints = np.array([
        np.zeros(3),
        np.ones(3),
        np.ones(3) * 2.0,
    ])

    trajectory = build_waypoint_trajectory(waypoints, segment_duration=1.0, hz=2.0)

    assert trajectory.shape == (5, 3)
    np.testing.assert_allclose(trajectory[0], waypoints[0])
    np.testing.assert_allclose(trajectory[-1], waypoints[-1])


def test_build_sine_demo_bounds_and_shape():
    trajectory = build_sine_demo(5, duration=1.0, hz=10.0)

    assert trajectory.shape == (11, 5)
    base = np.linspace(-0.6, 0.6, 5)
    amp = np.linspace(0.15, 0.30, 5)
    assert np.all(trajectory >= (base - amp - 1e-6))
    assert np.all(trajectory <= (base + amp + 1e-6))


def test_build_sine_trajectory_bounds_and_shape():
    trajectory = build_sine_trajectory(5, duration=1.0, hz=10.0)

    assert trajectory.shape == (11, 5)
    base = np.linspace(-0.6, 0.6, 5)
    amp = np.linspace(0.15, 0.30, 5)
    assert np.all(trajectory >= (base - amp - 1e-6))
    assert np.all(trajectory <= (base + amp + 1e-6))


def test_demo_waypoints_return_to_start():
    waypoints = demo_waypoints(5)

    assert waypoints.shape == (5, 5)
    np.testing.assert_allclose(waypoints[0], waypoints[-1])


def test_default_waypoints_return_to_start():
    waypoints = default_waypoints(5)

    assert waypoints.shape == (5, 5)
    np.testing.assert_allclose(waypoints[0], waypoints[-1])


def test_generate_demo_trajectory_returns_validated_object():
    trajectory = generate_demo_trajectory(4, mode="sine", hz=10.0, duration=1.0)

    assert isinstance(trajectory, Trajectory)
    assert trajectory.dof == 4
    assert trajectory.steps == 11
    assert trajectory.meta["demo_mode"] == "sine"


def test_generate_trajectory_returns_validated_object():
    trajectory = generate_trajectory(4, program="sine", hz=10.0, duration=1.0)

    assert isinstance(trajectory, Trajectory)
    assert trajectory.dof == 4
    assert trajectory.steps == 11
    assert trajectory.meta["program"] == "sine"


def test_available_programs_and_generate_positions():
    assert available_programs() == ("sine", "waypoints")
    waypoints = generate_positions(3, program="waypoints", hz=10.0, segment_duration=1.0)
    sine = generate_positions(3, program="sine", hz=10.0, duration=1.0)

    assert waypoints.shape[1] == 3
    assert sine.shape == (11, 3)


def test_resolve_record_destination_defaults_to_recordings(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("ei_vo.core.recording.time.strftime", lambda fmt: "20240102-030405")

    path, auto_dir = resolve_record_destination("")

    expected_dir = tmp_path / "recordings"
    assert path == (expected_dir / "demo_20240102-030405.mp4").as_posix()
    assert auto_dir == expected_dir.as_posix()


def test_resolve_record_destination_accepts_directory(tmp_path, monkeypatch):
    target_dir = tmp_path / "movies"
    target_dir.mkdir()
    monkeypatch.setattr("ei_vo.core.recording.time.strftime", lambda fmt: "20240102-030405")

    path, auto_dir = resolve_record_destination(target_dir)

    assert path == (target_dir / "demo_20240102-030405.mp4").as_posix()
    assert auto_dir == target_dir.as_posix()


def test_resolve_record_destination_preserves_filename(tmp_path):
    path, auto_dir = resolve_record_destination(tmp_path / "out")

    assert path == (tmp_path / "out").as_posix()
    assert auto_dir is None


def test_cli_wrapper_reexports_generic_entrypoints():
    cli = importlib.import_module("ei_vo.cli.playback")
    demo_cli = importlib.import_module("ei_vo.cli.demo")

    assert demo_cli.build_parser is cli.build_parser
    assert demo_cli.build_trajectory is cli.build_trajectory
    assert demo_cli.main is cli.main


def test_cli_main_builds_program_and_calls_play(monkeypatch):
    cli = importlib.import_module("ei_vo.cli.playback")
    calls = {}

    monkeypatch.setattr(cli.os.path, "isfile", lambda path: True)
    monkeypatch.setattr(cli, "_load_model_dof", lambda path: 3)

    def fake_play(model_path, trajectory, **kwargs):
        calls["model_path"] = model_path
        calls["trajectory"] = trajectory
        calls["kwargs"] = kwargs

    monkeypatch.setattr(cli, "play", fake_play)

    exit_code = cli.main(["--model", "robot.urdf", "--program", "waypoints", "--hz", "10"])

    assert exit_code == 0
    assert calls["model_path"] == "robot.urdf"
    assert isinstance(calls["trajectory"], Trajectory)
    assert calls["trajectory"].dof == 3
    assert calls["kwargs"]["hz"] == 10.0
    assert calls["kwargs"]["renderer"] == "matplotlib"
    assert calls["kwargs"]["kinematics"] is None
    assert calls["kwargs"]["record_path"] is None
    assert calls["kwargs"]["record_frames_dir"] is None
    assert calls["kwargs"]["camera"] is None


def test_cli_main_passes_camera_settings(monkeypatch):
    cli = importlib.import_module("ei_vo.cli.playback")
    calls = {}

    monkeypatch.setattr(cli.os.path, "isfile", lambda path: True)
    monkeypatch.setattr(cli, "_load_model_dof", lambda path: 3)

    def fake_play(model_path, trajectory, **kwargs):
        calls["model_path"] = model_path
        calls["trajectory"] = trajectory
        calls["kwargs"] = kwargs

    monkeypatch.setattr(cli, "play", fake_play)

    exit_code = cli.main(
        [
            "--renderer",
            "meshcat",
            "--model",
            "robot.urdf",
            "--program",
            "waypoints",
            "--cameraDistance",
            "3.5",
            "--cameraAzimuth",
            "120",
            "--cameraElevation",
            "-25",
            "--cameraLookat",
            "0.1",
            "0.2",
            "0.3",
        ]
    )

    assert exit_code == 0
    assert calls["model_path"] == "robot.urdf"
    assert isinstance(calls["trajectory"], Trajectory)
    assert calls["kwargs"]["renderer"] == "meshcat"
    assert calls["kwargs"]["camera"] == {
        "distance": 3.5,
        "azimuth": 120.0,
        "elevation": -25.0,
        "lookat": (0.1, 0.2, 0.3),
    }


def test_cli_main_loads_camera_file_and_applies_overrides(monkeypatch, tmp_path: pathlib.Path):
    cli = importlib.import_module("ei_vo.cli.playback")
    calls = {}
    camera_path = tmp_path / "front.camera.json"
    camera_path.write_text(
        json.dumps(
            {
                "distance": 2.5,
                "azimuth": 90.0,
                "elevation": -10.0,
                "lookat": [0.25, 0.5, 0.75],
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(cli.os.path, "isfile", lambda path: True)
    monkeypatch.setattr(cli, "_load_model_dof", lambda path: 3)

    def fake_play(model_path, trajectory, **kwargs):
        calls["model_path"] = model_path
        calls["trajectory"] = trajectory
        calls["kwargs"] = kwargs

    monkeypatch.setattr(cli, "play", fake_play)

    exit_code = cli.main(
        [
            "--renderer",
            "meshcat",
            "--model",
            "robot.urdf",
            "--program",
            "waypoints",
            "--cameraFile",
            str(camera_path),
            "--cameraElevation",
            "15",
        ]
    )

    assert exit_code == 0
    assert calls["kwargs"]["camera"] == {
        "distance": 2.5,
        "azimuth": 90.0,
        "elevation": 15.0,
        "lookat": (0.25, 0.5, 0.75),
    }


def test_cli_main_can_save_camera_preset(monkeypatch, tmp_path: pathlib.Path):
    cli = importlib.import_module("ei_vo.cli.playback")
    calls = {}
    output_path = tmp_path / "saved.camera.json"

    monkeypatch.setattr(cli.os.path, "isfile", lambda path: True)
    monkeypatch.setattr(cli, "_load_model_dof", lambda path: 3)

    def fake_play(model_path, trajectory, **kwargs):
        calls["model_path"] = model_path
        calls["trajectory"] = trajectory
        calls["kwargs"] = kwargs

    monkeypatch.setattr(cli, "play", fake_play)

    exit_code = cli.main(
        [
            "--renderer",
            "meshcat",
            "--model",
            "robot.urdf",
            "--program",
            "waypoints",
            "--cameraDistance",
            "4.0",
            "--cameraAzimuth",
            "30",
            "--cameraElevation",
            "5",
            "--cameraLookat",
            "1.0",
            "2.0",
            "3.0",
            "--saveCamera",
            str(output_path),
        ]
    )

    assert exit_code == 0
    assert calls["kwargs"]["camera"] == {
        "distance": 4.0,
        "azimuth": 30.0,
        "elevation": 5.0,
        "lookat": (1.0, 2.0, 3.0),
    }
    assert json.loads(output_path.read_text(encoding="utf-8")) == {
        "distance": 4.0,
        "azimuth": 30.0,
        "elevation": 5.0,
        "lookat": [1.0, 2.0, 3.0],
    }


def test_cli_meshcat_record_defaults_to_html(monkeypatch, tmp_path: pathlib.Path):
    cli = importlib.import_module("ei_vo.cli.playback")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("ei_vo.core.recording.time.strftime", lambda fmt: "20240102-030405")

    args = cli.build_parser().parse_args(["--renderer", "meshcat", "--record"])
    path, auto_dir = cli._resolve_recording(args)

    expected_dir = tmp_path / "recordings"
    assert path == (expected_dir / "meshcat_20240102-030405.html").as_posix()
    assert auto_dir == expected_dir.as_posix()


def test_cli_main_passes_record_frames_dir(monkeypatch):
    cli = importlib.import_module("ei_vo.cli.playback")
    calls = {}

    monkeypatch.setattr(cli.os.path, "isfile", lambda path: True)
    monkeypatch.setattr(cli, "_load_model_dof", lambda path: 3)

    def fake_play(model_path, trajectory, **kwargs):
        calls["model_path"] = model_path
        calls["trajectory"] = trajectory
        calls["kwargs"] = kwargs

    monkeypatch.setattr(cli, "play", fake_play)

    exit_code = cli.main(
        [
            "--renderer",
            "meshcat",
            "--model",
            "robot.urdf",
            "--program",
            "waypoints",
            "--record",
            "recordings/scene.mp4",
            "--recordFramesDir",
            "recordings/frames",
        ]
    )

    assert exit_code == 0
    assert calls["kwargs"]["record_path"] == "recordings/scene.mp4"
    assert calls["kwargs"]["record_frames_dir"] == "recordings/frames"


def test_cli_matplotlib_renderer_uses_model(monkeypatch):
    cli = importlib.import_module("ei_vo.cli.playback")
    calls = {}

    monkeypatch.setattr(cli.os.path, "isfile", lambda path: True)
    monkeypatch.setattr(cli, "_load_model_dof", lambda path: 2)

    def fake_play(model_path, trajectory, **kwargs):
        calls["model_path"] = model_path
        calls["trajectory"] = trajectory
        calls["kwargs"] = kwargs

    monkeypatch.setattr(cli, "play", fake_play)

    exit_code = cli.main(["--renderer", "matplotlib", "--model", "robot.urdf", "--program", "sine", "--hz", "20"])

    assert exit_code == 0
    assert calls["model_path"] == "robot.urdf"
    assert isinstance(calls["trajectory"], Trajectory)
    assert calls["trajectory"].dof == 2
    assert calls["kwargs"]["renderer"] == "matplotlib"


def test_view_cli_defaults_to_meshcat(monkeypatch):
    view = importlib.import_module("ei_vo.cli.view")
    calls = {}

    def fake_play(model_path, trajectory, **kwargs):
        calls["model_path"] = model_path
        calls["trajectory"] = trajectory
        calls["kwargs"] = kwargs

    monkeypatch.setattr(view.playback, "play", fake_play)

    exit_code = view.main(["--model", str(_EXAMPLE_MODEL)])

    assert exit_code == 0
    assert calls["model_path"] == str(_EXAMPLE_MODEL)
    assert isinstance(calls["trajectory"], Trajectory)
    assert calls["trajectory"].dof == 3
    assert calls["trajectory"].meta == {"mode": "view"}
    assert not np.allclose(calls["trajectory"].q, np.zeros((1, 3)))
    assert calls["kwargs"]["renderer"] == "meshcat"
    assert calls["kwargs"]["hold_open"] is True
    assert calls["kwargs"]["camera"]["distance"] > 0.0
    assert len(calls["kwargs"]["camera"]["lookat"]) == 3


def test_view_cli_enables_interactive_pyrender_and_saves_final_camera(monkeypatch, tmp_path: pathlib.Path):
    view = importlib.import_module("ei_vo.cli.view")
    calls = {}
    output_path = tmp_path / "saved.camera.json"
    final_camera = view.playback.CameraSettings(
        distance=3.0,
        azimuth=45.0,
        elevation=15.0,
        lookat=(1.0, 2.0, 3.0),
    )

    def fake_play(model_path, trajectory, **kwargs):
        calls["model_path"] = model_path
        calls["trajectory"] = trajectory
        calls["kwargs"] = kwargs
        return final_camera

    monkeypatch.setattr(view.playback, "play", fake_play)

    exit_code = view.main(
        [
            "--renderer",
            "pyrender",
            "--model",
            str(_EXAMPLE_MODEL),
            "--saveCamera",
            str(output_path),
        ]
    )

    assert exit_code == 0
    assert calls["kwargs"]["renderer"] == "pyrender"
    assert calls["kwargs"]["interactive"] is True
    assert json.loads(output_path.read_text(encoding="utf-8")) == {
        "distance": 3.0,
        "azimuth": 45.0,
        "elevation": 15.0,
        "lookat": [1.0, 2.0, 3.0],
    }


def test_view_cli_meshcat_save_camera_skips_hold_open(monkeypatch, tmp_path: pathlib.Path):
    view = importlib.import_module("ei_vo.cli.view")
    calls = {}
    output_path = tmp_path / "saved.camera.json"

    def fake_play(model_path, trajectory, **kwargs):
        calls["model_path"] = model_path
        calls["trajectory"] = trajectory
        calls["kwargs"] = kwargs

    monkeypatch.setattr(view.playback, "play", fake_play)

    exit_code = view.main(
        [
            "--model",
            str(_EXAMPLE_MODEL),
            "--cameraDistance",
            "4.0",
            "--saveCamera",
            str(output_path),
        ]
    )

    assert exit_code == 0
    assert "hold_open" not in calls["kwargs"]
    saved = json.loads(output_path.read_text(encoding="utf-8"))
    assert saved["distance"] == 4.0
    assert saved["azimuth"] is None
    assert saved["elevation"] is None
    assert isinstance(saved["lookat"], list)
    assert len(saved["lookat"]) == 3


@pytest.mark.parametrize(
    ("argv"),
    [
        ["--model", "robot.urdf", "--program", "waypoints"],
        ["--model", "robot.urdf", "--trajectries", "trajectory.csv"],
    ],
)
def test_view_cli_rejects_motion_arguments(argv):
    view = importlib.import_module("ei_vo.cli.view")

    with pytest.raises(SystemExit):
        view.main(argv)


def test_cli_accepts_trajectries_option(monkeypatch, tmp_path: pathlib.Path):
    cli = importlib.import_module("ei_vo.cli.playback")
    calls = {}
    csv_path = tmp_path / "trajectory.csv"
    np.savetxt(csv_path, np.array([[0.0, 1.0], [2.0, 3.0]], dtype=float), delimiter=",", fmt="%.6f")

    monkeypatch.setattr(cli.os.path, "isfile", lambda path: True)
    monkeypatch.setattr(cli, "_load_model_dof", lambda path: 2)

    def fake_play(model_path, trajectory, **kwargs):
        calls["model_path"] = model_path
        calls["trajectory"] = trajectory
        calls["kwargs"] = kwargs

    monkeypatch.setattr(cli, "play", fake_play)

    exit_code = cli.main(["--model", "robot.urdf", "--trajectries", str(csv_path), "--hz", "20"])

    assert exit_code == 0
    assert calls["model_path"] == "robot.urdf"
    assert isinstance(calls["trajectory"], Trajectory)
    assert calls["trajectory"].dof == 2
    assert calls["kwargs"]["renderer"] == "matplotlib"
    assert calls["kwargs"]["kinematics"] is None


def test_cli_accepts_backend(monkeypatch):
    cli = importlib.import_module("ei_vo.cli.playback")
    calls = {}

    monkeypatch.setattr(cli.os.path, "isfile", lambda path: True)
    monkeypatch.setattr(cli, "_load_model_dof", lambda path: 3)

    def fake_play(model_path, trajectory, **kwargs):
        calls["model_path"] = model_path
        calls["trajectory"] = trajectory
        calls["kwargs"] = kwargs

    monkeypatch.setattr(cli, "play", fake_play)

    exit_code = cli.main(
        [
            "--model",
            "robot.urdf",
            "--program",
            "waypoints",
            "--backend",
            "literobo",
            "--base-link",
            "base",
            "--end-link",
            "ee",
        ]
    )

    assert exit_code == 0
    assert calls["model_path"] == "robot.urdf"
    assert isinstance(calls["trajectory"], Trajectory)
    assert isinstance(calls["kwargs"]["kinematics"], KinematicsSpec)
    assert calls["kwargs"]["kinematics"].backend == "literobo"
    assert calls["kwargs"]["kinematics"].model_path == "robot.urdf"
    assert calls["kwargs"]["kinematics"].base_link == "base"
    assert calls["kwargs"]["kinematics"].end_link == "ee"


def test_cli_rejects_xml_model(monkeypatch):
    cli = importlib.import_module("ei_vo.cli.playback")
    monkeypatch.setattr(cli.os.path, "isfile", lambda path: True)

    with pytest.raises(ValueError, match="Only URDF models"):
        cli.main(["--model", "robot.xml", "--program", "waypoints"])


def test_top_level_play_is_lazy():
    assert callable(ei_vo.play)
