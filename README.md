# ei-vo

`ei-vo` is a small trajectory rendering and kinematics library built around a
shared URDF workflow. The base install keeps a lightweight `matplotlib`
renderer available, while `meshcat`, `pinocchio`, `literobo`, and `pyrender`
stay optional.

## Setup

```bash
uv sync
# add as needed:
#   uv sync --extra meshcat
#   uv sync --extra kinematics
#   uv sync --extra pyrender
```

## Quick Start

This README uses `matplotlib` as the baseline renderer because it works in the
base install and requires only a URDF model.

Play a built-in trajectory:

```bash
uv run ei-vo-play \
  --renderer matplotlib \
  --model examples/models/three_dof_arm.urdf \
  --program waypoints \
  --hz 120
```

Replay a trajectory file:

```bash
uv run ei-vo-play \
  --renderer matplotlib \
  --model examples/models/three_dof_arm.urdf \
  --trajectries examples/trajectories/three_dof_arm_waypoints.csv
```

Save the final frame:

```bash
uv run ei-vo-play \
  --renderer matplotlib \
  --model examples/models/three_dof_arm.urdf \
  --program waypoints \
  --record recordings/link_frame.png
```

Record the same playback to MP4:

```bash
brew install ffmpeg

uv run ei-vo-play \
  --renderer matplotlib \
  --model examples/models/three_dof_arm.urdf \
  --program waypoints \
  --record recordings/link_motion.mp4
```

The same `--model` and trajectory inputs work across the built-in renderers.
No kinematics backend is selected by default.

## Switch Renderer

In most cases, changing renderer just means changing `--renderer`.

| Renderer | Install | Typical use | Output |
| --- | --- | --- | --- |
| `matplotlib` | base install | simplest baseline viewer/export | live figure, `.png`, `.mp4` |
| `meshcat` | `uv sync --extra meshcat` | browser playback with URDF visuals | live viewer, standalone `.html` |
| `pyrender` | `uv sync --extra pyrender` | headless URDF offscreen export | `.png`, `.mp4` |

MeshCat:

```bash
uv sync --extra meshcat
uv run ei-vo-play --renderer meshcat --model examples/models/three_dof_arm.urdf --program waypoints
```

Pyrender:

```bash
uv sync --extra pyrender
uv run ei-vo-play --renderer pyrender --model examples/models/three_dof_arm.urdf --program waypoints --record recordings/pyrender.mp4
```

If you need a kinematics backend in the same workflow:

```bash
uv sync --extra kinematics

uv run ei-vo-play \
  --renderer meshcat \
  --model examples/models/three_dof_arm.urdf \
  --backend literobo \
  --base-link base \
  --end-link ee
```

MeshCat uses Pinocchio to display URDF `<visual>` geometry. Only URDF input is
supported by the CLI.

## Recording Notes

- `matplotlib`: `.png` or `.mp4`
- `meshcat`: standalone `.html`
- `pyrender`: `.png` or `.mp4`, and `--record` is required
- `--recordFramesDir` is supported for Matplotlib and Pyrender video export

MeshCat HTML export:

```bash
uv run ei-vo-play \
  --renderer meshcat \
  --model examples/models/three_dof_arm.urdf \
  --trajectries examples/trajectories/three_dof_arm_waypoints.csv \
  --record recordings/scene.html
```

Pyrender still image:

```bash
uv run ei-vo-play \
  --renderer pyrender \
  --model examples/models/three_dof_arm.urdf \
  --program waypoints \
  --record recordings/pyrender_frame.png \
  --recordSize 1280 720
```

## Notes

- Base install depends on `numpy` and `matplotlib`.
- Optional extras are `pinocchio`, `meshcat`, `kinematics`, `literobo`, and `pyrender`.
- URDF metadata such as DOF and joint limits is parsed without optional backends.
- The built-in renderers accept URDF only.
- Pyrender uses `urdfpy` under a small Python 3.13 / NumPy 2 compatibility shim because upstream still pins an older `networkx`.
- Pyrender is offscreen-only in `ei-vo`; it requires `--record` and does not support `--loop`.
- Pyrender offscreen rendering may require `PYOPENGL_PLATFORM=egl` or `PYOPENGL_PLATFORM=osmesa` on headless Linux hosts.
- Video export for Pyrender or Matplotlib needs `ffmpeg` on `PATH` or `EI_VO_FFMPEG`.
- Input files for `--trajectries` must be shaped `(T, DOF)` and may be CSV, NPY, or JSON.

## Python API

The Python API mirrors the CLI through `render_program`, `render_trajectory`,
and `trajectory_from_program`.

```python
from pathlib import Path
from ei_vo import RenderSpec, render_program

render_program(
    Path("examples/models/three_dof_arm.urdf"),
    program="waypoints",
    renderer=RenderSpec("matplotlib", options={"show": False}),
    record_path="recordings/link_frame.png",
)
```

## Compatibility

```bash
uv run ei-vo-demo --renderer matplotlib --model examples/models/three_dof_arm.urdf --demo wp
uv run python examples/switch_renderer.py
uv run pytest
```
