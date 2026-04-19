# ei-vo

`ei-vo` is a small trajectory rendering and kinematics library built around a
shared URDF workflow. `pinocchio`, `matplotlib`, and `mujoco` are part of the
standard install, and this README assumes `pinocchio` as the default
kinematics backend.

## Setup

```bash
uv sync
# add as needed: --extra meshcat --extra recording
```

## Quick Start

This README uses `matplotlib` as the baseline renderer because it is the
simplest way to understand the workflow.

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
uv sync --extra recording
brew install ffmpeg

uv run ei-vo-play \
  --renderer matplotlib \
  --model examples/models/three_dof_arm.urdf \
  --program waypoints \
  --record recordings/link_motion.mp4
```

The same `--model` and trajectory inputs work across the built-in renderers.
`pinocchio` is the default backend.

## Switch Renderer

In most cases, changing renderer just means changing `--renderer`.

| Renderer | Typical use | Output |
| --- | --- | --- |
| `matplotlib` | simplest baseline | live figure, `.png`, `.mp4` |
| `meshcat` | browser playback | live viewer, standalone `.html` |
| `mujoco` | interactive desktop viewer | live viewer, `.mp4` |
| `blender` | offline rendering | `.png`, `.mp4` |

MeshCat:

```bash
uv run ei-vo-play --renderer meshcat --model examples/models/three_dof_arm.urdf --program waypoints
```

MuJoCo:

```bash
uv run ei-vo-play --renderer mujoco --model examples/models/three_dof_arm.urdf --program waypoints
```

Blender:

```bash
uv run ei-vo-play --renderer blender --model examples/models/three_dof_arm.urdf --program waypoints --record recordings/blender.mp4
```

If you need a kinematics backend in the same workflow:

```bash
uv run ei-vo-play \
  --renderer meshcat \
  --model examples/models/three_dof_arm.urdf \
  --backend literobo \
  --base-link base \
  --end-link ee
```

MeshCat uses Pinocchio to display URDF `<visual>` geometry. XML/MJCF input is
not supported in the CLI.

## Recording Notes

- `matplotlib`: `.png` or `.mp4`
- `mujoco`: `.mp4`
- `meshcat`: standalone `.html`
- `blender`: `.png` or `.mp4`, and `--record` is required
- `--recordFramesDir` is supported for Blender, MuJoCo, and Matplotlib video export

MeshCat HTML export:

```bash
uv run ei-vo-play \
  --renderer meshcat \
  --model examples/models/three_dof_arm.urdf \
  --trajectries examples/trajectories/three_dof_arm_waypoints.csv \
  --record recordings/scene.html
```

Blender preview render:

```bash
uv run ei-vo-play \
  --renderer blender \
  --model examples/models/three_dof_arm.urdf \
  --program waypoints \
  --record recordings/blender_preview.mp4 \
  --recordFps 24 \
  --recordSize 960 540 \
  --blenderEngine workbench \
  --blenderSamples 1
```

## Notes

- Base install only depends on `numpy`.
- URDF metadata such as DOF and joint limits is parsed without MuJoCo.
- Blender prefers a local `blender` executable on `PATH` or `EI_VO_BLENDER`.
- Blender can fall back to `bpy`; install it with `uv add bpy`.
- Video export for Blender, MuJoCo, or Matplotlib needs `ffmpeg` on `PATH` or `EI_VO_FFMPEG`.
- On macOS, MuJoCo interactive playback requires `mjpython`. The CLI relaunches automatically for `--renderer mujoco`.
- Blender scene cache is stored under `$TMPDIR/ei_vo_blender_cache`.
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
uv run ei-vo-demo --renderer mujoco --model examples/models/three_dof_arm.urdf --demo wp
uv run python examples/demo_mj.py --renderer mujoco --model examples/models/three_dof_arm.urdf --demo wp
uv run python examples/switch_renderer.py
uv run pytest
```
