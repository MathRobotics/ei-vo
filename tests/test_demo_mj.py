import json
import math
import pathlib
import sys
import types

import numpy as np

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

    assert loaded.shape == (2, 7)
    expected = np.deg2rad(data[:, :7])
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

    assert loaded.shape == (2, 7)
    np.testing.assert_allclose(loaded, arr[:, :7])


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
    traj = demo_mj.build_sine_demo(T_sec=1.0, hz=10.0)

    assert traj.shape == (11, 7)

    base = np.array([0.0, -0.6, 0.0, -1.8, 0.0, 1.4, 0.6])
    amp = np.array([0.25, 0.15, 0.20, 0.25, 0.20, 0.20, 0.15])
    lower = base - amp - 1e-6
    upper = base + amp + 1e-6
    assert np.all(traj >= lower)
    assert np.all(traj <= upper)
