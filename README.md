# ei-vo

`ei-vo` is a small trajectory rendering and kinematics library with pluggable
backends. It ships with MuJoCo, MeshCat, Matplotlib, Pinocchio, and LiteRobo
integration points.

## Setup

```bash
pip install -e .[dev]
```

- `mujoco` backend: requires MuJoCo on the host system
- `meshcat` backend: requires `meshcat` and a URDF model
- `matplotlib` backend: requires `matplotlib`, `mujoco`, and a URDF model
- `pinocchio` kinematics backend: requires the `pin` package
- `literobo` kinematics backend: requires `literobo` and a URDF model
- MP4 recording with MuJoCo additionally uses `imageio[ffmpeg]`
- On macOS, MuJoCo's passive viewer requires `mjpython`. The CLI relaunches
  `--renderer mujoco` runs automatically via `mjpython`; direct Python scripts
  that use the MuJoCo renderer should be started with `mjpython`.

## Layout

- `ei_vo.core`: validated trajectory/model types and file-loading helpers
- `ei_vo.programs`: built-in waypoint and sine trajectory programs
- `ei_vo.demo`: compatibility aliases for the legacy demo naming
- `ei_vo.backends`: renderer and kinematics selection helpers
- `ei_vo.workflows`: high-level helpers for copy-paste friendly examples
- `ei_vo.kinematics`: pluggable forward-kinematics backends
- `ei_vo.render.registry`: lazy backend registration and dispatch
- `ei_vo.render.render_mj`: MuJoCo playback, recording, and model inspection
- `ei_vo.render.render_meshcat`: browser-based 3D playback through MeshCat
- `ei_vo.render.render_matplotlib`: Matplotlib-based 3D playback for MuJoCo models
- `ei_vo.cli.playback`: reusable CLI entrypoint
- `ei_vo.cli.demo`: compatibility alias for the legacy module name
- `ei_vo.cli.demo_mj`: compatibility alias for the legacy module name
- `examples/demo_mj.py`: thin compatibility wrapper around the CLI
- `examples/switch_renderer.py`: single Python example using the high-level workflow API

## CLI usage

The package exposes a single CLI with selectable renderers and optional backend
wiring:

```bash
ei-vo-play --model examples/models/three_dof_arm.urdf --program waypoints --hz 240
```

This uses `meshcat` for rendering and `literobo` for the optional kinematics
backend by default.

Matplotlib 3D playback using the same URDF model:

```bash
ei-vo-play --renderer matplotlib --model examples/models/three_dof_arm.urdf --hz 120
```

Browser-based 3D playback through MeshCat:

```bash
ei-vo-play \
  --renderer meshcat \
  --model examples/models/three_dof_arm.urdf \
  --trajectries examples/trajectories/three_dof_arm_waypoints.csv
```

Attach a backend to the same playback workflow:

```bash
ei-vo-play \
  --renderer meshcat \
  --model examples/models/three_dof_arm.urdf \
  --backend literobo \
  --base-link base \
  --end-link ee
```

The bundled `mujoco`, `meshcat`, and `matplotlib` renderers all read the same
URDF passed via `--model`. `--backend` selects the workflow's kinematics
backend. XML/MJCF input is no longer supported in the CLI.

Compatibility aliases still work:

```bash
ei-vo-demo --renderer mujoco --model examples/models/three_dof_arm.urdf --demo wp
python examples/demo_mj.py --renderer mujoco --model examples/models/three_dof_arm.urdf --demo wp
```

On macOS, these MuJoCo CLI commands are relaunched through `mjpython`
automatically.

Replay an existing trajectory file:

```bash
ei-vo-play \
  --model examples/models/three_dof_arm.urdf \
  --trajectries examples/trajectories/three_dof_arm_waypoints.csv
```

Record MuJoCo playback to MP4:

```bash
ei-vo-play \
  --model examples/models/three_dof_arm.urdf \
  --record recordings/playback.mp4 \
  --recordFps 60 \
  --recordSize 1920 1080
```

Save the final Matplotlib frame:

```bash
ei-vo-play \
  --renderer matplotlib \
  --model examples/models/three_dof_arm.urdf \
  --trajectries examples/trajectories/three_dof_arm_waypoints.csv \
  --record recordings/link_frame.png
```

Export a MeshCat scene as HTML:

```bash
ei-vo-play \
  --renderer meshcat \
  --model examples/models/three_dof_arm.urdf \
  --trajectries examples/trajectories/three_dof_arm_waypoints.csv \
  --record recordings/scene.html
```

## Python usage

```python
from pathlib import Path

from ei_vo import (
    KinematicsSpec,
    RenderSpec,
    render_program,
    render_trajectory,
    trajectory_from_program,
)

ROOT = Path("examples")

trajectory = trajectory_from_program(3, program="sine", hz=120.0, duration=4.0)

render_program(ROOT / "models/three_dof_arm.urdf", program="waypoints", hz=240.0)
render_program(
    ROOT / "models/three_dof_arm.urdf",
    hz=120.0,
    renderer=RenderSpec("matplotlib", options={"show": False, "title": "Matplotlib 3D Playback"}),
    record_path="link_frame.png",
)
render_program(
    ROOT / "models/three_dof_arm.urdf",
    renderer="mujoco",
    kinematics=KinematicsSpec(
        "pinocchio",
        base_link="base",
        end_link="ee",
    ),
)
render_trajectory(ROOT / "models/three_dof_arm.urdf", trajectory, renderer=RenderSpec("meshcat"))
```

On macOS, scripts that call the Python API with `renderer="mujoco"` should be
started with `mjpython`.

## Python example

- `python examples/switch_renderer.py`
- edit `MODEL`, `RENDERER`, and `BACKEND` in `examples/switch_renderer.py`
- if you switch `RENDERER` to `"mujoco"` on macOS, the script relaunches
  itself via `mjpython`

## Sample assets

- `examples/models/three_dof_arm.urdf`
- `examples/trajectories/three_dof_arm_waypoints.csv`

## Tests

```bash
pytest
```
