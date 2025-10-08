import json
import math
import pathlib
import sys
import types

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

dummy_ei = types.ModuleType("ei")
dummy_ei.play = lambda *args, **kwargs: None
sys.modules.setdefault("ei", dummy_ei)

from examples import demo_mj


def test_load_angles_csv_in_degrees(tmp_path: pathlib.Path):
    data = np.array([
        [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 999.0],
        [90.0, 80.0, 70.0, 60.0, 50.0, 40.0, 30.0, -999.0],
    ], dtype=float)
    csv_path = tmp_path / "angles.csv"
    np.savetxt(csv_path, data, delimiter=",", fmt="%.6f")

    loaded = demo_mj.load_angles(str(csv_path), deg=True)

    assert loaded.shape == (2, 8)
    expected = np.deg2rad(data)
    np.testing.assert_allclose(loaded, expected)


def test_load_angles_json_truncates_columns(tmp_path: pathlib.Path):
    arr = np.array([
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    ])
    json_path = tmp_path / "angles.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(arr.tolist(), f)

    loaded = demo_mj.load_angles(str(json_path), deg=False)

    assert loaded.shape == (2, 8)
    np.testing.assert_allclose(loaded, arr)


def test_quintic_interpolation_hits_endpoints():
    q0 = np.zeros(7)
    q1 = np.ones(7) * math.pi
    segment = demo_mj.quintic(q0, q1, T=1.0, dt=0.2)

    assert np.allclose(segment[0], q0)
    assert np.allclose(segment[-1], q1)
    assert segment.shape[0] == 6  # inclusive of both endpoints (0.0 .. 1.0 step 0.2)
    assert np.all(segment[1:] >= segment[:-1] - 1e-9)


def test_build_demo_trajectory_concatenates_segments():
    waypoints = np.array([
        np.zeros(7),
        np.ones(7),
        np.ones(7) * 2,
    ])
    traj = demo_mj.build_demo_trajectory(waypoints, seg_T=1.0, hz=2.0)

    # With hz=2 the dt is 0.5, so each quintic segment yields 3 samples.
    # The function drops the last sample of each segment except the final waypoint.
    assert traj.shape[0] == 5
    np.testing.assert_allclose(traj[0], waypoints[0])
    np.testing.assert_allclose(traj[-1], waypoints[-1])


def test_build_sine_demo_bounds_and_shape():
    traj = demo_mj.build_sine_demo(7, T_sec=1.0, hz=10.0)

    assert traj.shape == (11, 7)

    base = np.linspace(-0.6, 0.6, 7)
    amp = np.linspace(0.15, 0.30, 7)
    lower = base - amp - 1e-6
    upper = base + amp + 1e-6
    assert np.all(traj >= lower)
    assert np.all(traj <= upper)


def test_demo_waypoints_matches_dof():
    wp = demo_mj.demo_waypoints(5)

    assert wp.shape == (5, 5)
    np.testing.assert_allclose(wp[0], wp[-1])


def test_three_dof_example_model_matches_dof():
    model_path = ROOT / "examples" / "models" / "three_dof_arm.xml"

    mj_model = demo_mj.mj.MjModel.from_xml_path(str(model_path))
    qaddrs = demo_mj.render_mj.detect_arm_joint_qaddr(mj_model)

    assert len(qaddrs) == 3

    wp = demo_mj.demo_waypoints(len(qaddrs))
    traj = demo_mj.build_demo_trajectory(wp, seg_T=0.5, hz=10.0)

    assert wp.shape[1] == len(qaddrs)
    assert traj.shape[1] == len(qaddrs)


def test_three_dof_sample_angles_match_model():
    model_path = ROOT / "examples" / "models" / "three_dof_arm.xml"
    traj_path = ROOT / "examples" / "trajectories" / "three_dof_arm_waypoints.csv"

    assert traj_path.is_file()

    mj_model = demo_mj.mj.MjModel.from_xml_path(str(model_path))
    qaddrs = demo_mj.render_mj.detect_arm_joint_qaddr(mj_model)

    angles = demo_mj.load_angles(str(traj_path), deg=False)

    assert len(qaddrs) == 3
    assert angles.shape[1] == len(qaddrs)
    np.testing.assert_allclose(angles[0], angles[-1], atol=1e-6)


def test_prepare_play_invocation_skips_record_kwargs_when_not_supported():
    original_play = demo_mj.play

    def stub_play(model_path, traj, slow=1.0, hz=240.0, loop=False):
        return None

    demo_mj.play = stub_play
    try:
        args = types.SimpleNamespace(
            model="model.xml",
            slow=0.5,
            hz=120.0,
            loop=True,
            record=None,
            recordFps=None,
            recordSize=None,
        )
        traj = types.SimpleNamespace(q=np.zeros((1, 7)))

        call_args, call_kwargs = demo_mj._prepare_play_invocation(args, traj)

        assert call_args == ["model.xml"]
        assert call_kwargs["traj"] is traj
        assert call_kwargs["slow"] == args.slow
        assert call_kwargs["hz"] == args.hz
        assert call_kwargs["loop"] is True
        assert "record_path" not in call_kwargs
    finally:
        demo_mj.play = original_play


def test_prepare_play_invocation_warns_when_record_not_supported():
    original_play = demo_mj.play

    def stub_play(model_path, traj, slow=1.0, hz=240.0, loop=False):
        return None

    demo_mj.play = stub_play
    try:
        args = types.SimpleNamespace(
            model="model.xml",
            slow=1.0,
            hz=240.0,
            loop=False,
            record="out.mp4",
            recordFps=30.0,
            recordSize=(640, 480),
        )
        traj = types.SimpleNamespace(q=np.zeros((1, 7)))

        with pytest.warns(RuntimeWarning) as record:
            demo_mj._prepare_play_invocation(args, traj)

        messages = {str(w.message) for w in record}
        assert any("record_path" in msg for msg in messages)
    finally:
        demo_mj.play = original_play


def test_prepare_play_invocation_includes_record_kwargs_when_supported():
    original_play = demo_mj.play

    def stub_play(model_path, traj, slow=1.0, hz=240.0, loop=False,
                  record_path=None, record_fps=None, record_size=None):
        return None

    demo_mj.play = stub_play
    try:
        args = types.SimpleNamespace(
            model="model.xml",
            slow=1.5,
            hz=120.0,
            loop=False,
            record="output",
            recordFps=60.0,
            recordSize=[800, 600],
        )
        traj = types.SimpleNamespace(q=np.zeros((1, 7)))

        call_args, call_kwargs = demo_mj._prepare_play_invocation(args, traj)

        assert call_args == ["model.xml"]
        assert call_kwargs["record_path"] == "output"
        assert call_kwargs["record_fps"] == 60.0
        assert call_kwargs["record_size"] == (800, 600)
    finally:
        demo_mj.play = original_play


def test_resolve_record_destination_defaults_to_recordings(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(demo_mj.time, "strftime", lambda fmt: "20240102-030405")

    path, auto_dir = demo_mj._resolve_record_destination("")

    expected_dir = tmp_path / "recordings"
    expected_file = expected_dir / "demo_20240102-030405.mp4"

    assert path == expected_file.as_posix()
    assert auto_dir == expected_dir.as_posix()


def test_resolve_record_destination_accepts_directory(tmp_path, monkeypatch):
    target_dir = tmp_path / "movies"
    target_dir.mkdir()
    monkeypatch.setattr(demo_mj.time, "strftime", lambda fmt: "20240102-030405")

    path, auto_dir = demo_mj._resolve_record_destination(str(target_dir))

    assert path == (target_dir / "demo_20240102-030405.mp4").as_posix()
    assert auto_dir == target_dir.as_posix()


def test_resolve_record_destination_preserves_filename(tmp_path):
    given = tmp_path / "out"

    path, auto_dir = demo_mj._resolve_record_destination(given)

    assert path == given.as_posix()
    assert auto_dir is None
