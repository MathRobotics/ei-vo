# Examples

This directory contains demo scripts built on MuJoCo models for 7-DOF robots.
It explains how to use `demo_mj.py`, prepare trajectories, and record videos.

Training-friendly MJCFs are included, such as `examples/models/simple_model.xml`
and the 3-DOF arm used in tests (`examples/models/three_dof_arm.xml`). If you
do not have your own model, point `--model` to one of these to try the demo
immediately. A ready-to-use reference trajectory is also provided for the
3-DOF model (`examples/trajectories/three_dof_arm_waypoints.csv`).

## Prerequisites

- MuJoCo 2.x installed, with environment variables like `LD_LIBRARY_PATH` and
  `MUJOCO_PY_MJKEY_PATH` set
- An MJCF (`model.xml`) on hand, or use `examples/models/simple_model.xml`
- Python 3.9+ and the dependencies listed in `requirements.txt`

## Running the demo

When no angle file is supplied, the script generates and plays a built-in demo
trajectory. The following runs the waypoint demo in real time:

```bash
python examples/demo_mj.py --model /path/to/model.xml
# Or with the bundled simple model
python examples/demo_mj.py --model examples/models/simple_model.xml
```

### Options

| Option | Description |
| --- | --- |
| `--angles PATH` | Load a joint angle file in CSV / NPY / JSON (`shape=(T, DOF)`) |
| `--deg` | Use when the angle file is in degrees (converted to radians) |
| `--hz FLOAT` | Playback frequency in Hz (default: 240.0) |
| `--loop` | Loop playback |
| `--demo {wp,sine}` | Switch demo trajectory when no angle file is supplied |
| `--segT FLOAT` | Segment duration [s] for the waypoint demo (default: 1.5) |
| `--slow FLOAT` | Slow-motion playback (`>1` plays slower) |
| `--record [PATH]` | Save a recording (e.g., `output.mp4`). When the filename is omitted, saves under `recordings/` |
| `--recordFps FLOAT` | Override the recording frame rate |
| `--recordSize W H` | Recording size in pixels (width, height) |

## Recording playback

Use `--record` to enable recording. Combine with `--recordFps` and
`--recordSize` to control the frame rate and resolution:

```bash
python examples/demo_mj.py \
    --model /path/to/model.xml \
    --record recordings/demo.mp4 \
    --recordFps 60 \
    --recordSize 1920 1080
```

If you pass `--record` without a filename, the video is saved to
`recordings/demo_<timestamp>.mp4` in the current directory. The viewer remains
visible during recording and the file is written when playback ends.

> **Note**
> If you see `Recording was requested but the installed ei.play() does not accept a 'record_path' argument.`, the `ei` package
> you are using is too old and lacks recording support. Run `pip install -e .`
> in the repository root or upgrade to the latest version.

## Preparing trajectory files

Files loaded via `--angles` are 2D arrays with columns equal to the model DOF.
Example for generating a CSV with NumPy:

```python
import numpy as np

# shape = (T, 7)
angles = np.linspace(0, 1, 240)[:, None] * np.ones((1, 7))
np.savetxt("traj.csv", angles, delimiter=",")
```

JSON and NPY formats are handled similarly. If your angles are in degrees, pass
`--deg`. For a quick 3-DOF trial, feed
`examples/trajectories/three_dof_arm_waypoints.csv` to `--angles` to replay the
smooth waypoint-based trajectory.

## Tests

`tests/` contains pytest-based checks for loading and recording behavior:

```bash
pytest
```

Stubs are provided so MuJoCo's native libraries are not required to run the
tests locally.
