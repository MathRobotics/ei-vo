# ei-vo

`ei-vo` is a small trajectory rendering and kinematics library built around a
shared URDF workflow. The base install keeps a lightweight `matplotlib`
renderer available, while `meshcat`, `literobo`, and `pyrender`
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

Play the bundled trajectory file:

```bash
uv run ei-vo-play \
  --renderer matplotlib \
  --model examples/models/three_dof_arm.urdf \
  --trajectries examples/trajectories/three_dof_arm_waypoints.csv \
  --hz 120
```

Play a built-in trajectory:

```bash
uv run ei-vo-play \
  --renderer matplotlib \
  --model examples/models/three_dof_arm.urdf
```

Omitting `--trajectries` falls back to the default built-in `waypoints`
program.

Save the final frame:

```bash
uv run ei-vo-play \
  --renderer matplotlib \
  --model examples/models/three_dof_arm.urdf \
  --trajectries examples/trajectories/three_dof_arm_waypoints.csv \
  --record recordings/link_frame.png
```

Record the same playback to MP4:

```bash
brew install ffmpeg

uv run ei-vo-play \
  --renderer matplotlib \
  --model examples/models/three_dof_arm.urdf \
  --trajectries examples/trajectories/three_dof_arm_waypoints.csv \
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
| `pyrender` | `uv sync --extra pyrender` | desktop viewer or offscreen URDF export | live viewer, `.png`, `.mp4` |

MeshCat:

```bash
uv sync --extra meshcat
uv run ei-vo-play --renderer meshcat --model examples/models/three_dof_arm.urdf --trajectries examples/trajectories/three_dof_arm_waypoints.csv
uv run ei-vo-view --model examples/models/three_dof_arm.urdf
```

Pyrender:

```bash
uv sync --extra pyrender
uv run ei-vo-play --renderer pyrender --model examples/models/three_dof_arm.urdf --trajectries examples/trajectories/three_dof_arm_waypoints.csv --record recordings/pyrender.mp4
uv run ei-vo-view --renderer pyrender --model examples/models/three_dof_arm.urdf
```

If you need a kinematics backend in the same workflow:

```bash
uv sync --extra kinematics

uv run ei-vo-play \
  --renderer meshcat \
  --model examples/models/three_dof_arm.urdf \
  --trajectries examples/trajectories/three_dof_arm_waypoints.csv \
  --backend literobo \
  --base-link base \
  --end-link ee
```

MeshCat uses `urdfpy` to display URDF `<visual>` geometry. Only URDF input is
supported by the CLI.

## Reuse Camera Presets

Use `ei-vo-view` when you want to inspect the URDF model itself and adjust the
camera. It renders a single zero-joint pose with `meshcat` selected by default:

```bash
uv run ei-vo-view \
  --model examples/models/three_dof_arm.urdf
```

Use `ei-vo-play` when you want to watch motion. `ei-vo-view` does not accept
trajectory or built-in program options.
With the default `meshcat` renderer, `ei-vo-view` keeps the viewer server alive
until you stop it with `Ctrl+C`.

Adjust the camera in the browser, then use the MeshCat GUI panel
`Save / Load / Capture -> save_scene` to download `scene.json`.

You can reuse that saved view directly on the next run:

```bash
uv run ei-vo-view \
  --model examples/models/three_dof_arm.urdf \
  --cameraFile ~/Downloads/scene.json
```

If you want a dedicated reusable preset, convert the resolved view into a small
camera JSON:

```bash
uv run ei-vo-view \
  --model examples/models/three_dof_arm.urdf \
  --cameraFile ~/Downloads/scene.json \
  --saveCamera recordings/front.camera.json
```

After that, the same preset works with `ei-vo-view` or `ei-vo-play`:

```bash
uv run ei-vo-play \
  --renderer meshcat \
  --model examples/models/three_dof_arm.urdf \
  --trajectries examples/trajectories/three_dof_arm_waypoints.csv \
  --cameraFile recordings/front.camera.json
```

For `pyrender`, `ei-vo-view` can save the final window camera directly when you
close the viewer:

```bash
uv run ei-vo-view \
  --renderer pyrender \
  --model examples/models/three_dof_arm.urdf \
  --saveCamera recordings/pyrender_view.camera.json
```

Move the camera in the viewer, close the window, and the final view is written
to the preset file.

## Recording Notes

- `matplotlib`: `.png` or `.mp4`
- `meshcat`: standalone `.html`
- `pyrender`: `ei-vo-play` writes `.png` or `.mp4`, and `ei-vo-view --renderer pyrender` opens a live window
- `--recordFramesDir` is supported for Matplotlib and Pyrender video export
- `--cameraFile` accepts either an `ei-vo` camera preset JSON or a MeshCat `scene.json`
- `ei-vo-view` is for static model inspection; `ei-vo-play` is for trajectory playback

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
  --trajectries examples/trajectories/three_dof_arm_waypoints.csv \
  --record recordings/pyrender_frame.png \
  --recordSize 1280 720
```

## Notes

- Base install depends on `numpy` and `matplotlib`.
- Optional extras are `meshcat`, `kinematics`, `literobo`, and `pyrender`.
- URDF metadata such as DOF and joint limits is parsed without optional backends.
- The built-in renderers accept URDF only.
- MeshCat and Pyrender use `urdfpy` under a small Python 3.13 / NumPy 2 compatibility shim because upstream still pins an older `networkx`.
- `ei-vo-play --renderer pyrender` is offscreen export and still requires `--record`.
- `ei-vo-view --renderer pyrender` opens the live Pyrender viewer.
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
