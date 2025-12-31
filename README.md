# ei-vo

`ei-vo` is a collection of MuJoCo-based visualization and teleoperation demos.
Using `examples/demo_mj.py` you can generate and replay demonstration
trajectories tailored to any robot model.

## Setup

```bash
pip install -e .
```

You need the MuJoCo runtime available on your system. Refer to the
[official documentation](https://mujoco.readthedocs.io/) for installation
instructions and environment variables such as `MUJOCO_PY_MJKEY_PATH`.

## MuJoCo demo (`examples/demo_mj.py`)

- Specify an MJCF file with `--model`.
- Provide a CSV/NPY/JSON trajectory file via `--angles` (shape=(T, DOF)) to
  replay it directly.
- If no trajectory is supplied, the script auto-detects the model DOF and
  generates a safe demo trajectory (waypoints or sine wave).
- Use `--record` to save playback as MP4 (accepts a file path or directory).
- `--recordFps` / `--recordSize` control the recording frame rate and
  resolution.

Example: play a waypoint demo for a 3-DOF model:

```bash
python examples/demo_mj.py \
  --model examples/models/three_dof_arm.xml \
  --demo wp \
  --hz 240
```

Using an angle file instead:

```bash
python examples/demo_mj.py \
  --model examples/models/three_dof_arm.xml \
  --angles my_angles.csv \
  --deg  # Use when the CSV is in degrees
```

## Sample models

- `examples/models/three_dof_arm.xml`: 3-DOF arm for testing and demos.
- `examples/trajectories/three_dof_arm_waypoints.csv`: reference trajectory
  (radians) for the model above; pass it to `--angles` to replay immediately.

## Tests

```bash
pytest
```

## License

See `pyproject.toml` for license details.
