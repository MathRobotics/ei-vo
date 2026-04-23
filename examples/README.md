# Examples

The main Python example is:

- `examples/switch_renderer.py`: switch the renderer and backend by editing variables

Bundled assets:

- `examples/models/three_dof_arm.urdf`
- `examples/trajectories/three_dof_arm_waypoints.csv`

## Running playback

```bash
uv run ei-vo-play --model examples/models/three_dof_arm.urdf --program waypoints
uv sync --extra meshcat
uv run ei-vo-play --renderer meshcat --model examples/models/three_dof_arm.urdf --program waypoints
uv run ei-vo-play --renderer matplotlib --model examples/models/three_dof_arm.urdf
uv sync --extra pyrender
uv run ei-vo-play --renderer pyrender --model examples/models/three_dof_arm.urdf --program waypoints --record recordings/pyrender.mp4
uv sync --extra pinocchio
uv run ei-vo-play --renderer meshcat --model examples/models/three_dof_arm.urdf --backend pinocchio --base-link base --end-link ee
```

The first command uses the default `matplotlib` renderer and does not attach a
kinematics backend.

## Running the Python example

```bash
uv run python examples/switch_renderer.py
```

Edit `MODEL`, `RENDERER`, and `BACKEND` to switch the playback setup. The file
defaults to `matplotlib` without a kinematics backend.

## Options

| Option | Description |
| --- | --- |
| `--renderer {matplotlib,meshcat,pyrender}` | Select the renderer backend |
| `--backend {literobo,pinocchio}` | Attach a kinematics backend to the playback workflow |
| `--base-link NAME` | Base link for the selected backend |
| `--end-link NAME` | End link for the selected backend |
| `--model PATH` | URDF model file. Required for the built-in renderers |
| `--trajectries PATH` | Load a trajectory from CSV / NPY / JSON |
| `--deg` | Convert `--trajectries` input from degrees to radians |
| `--hz FLOAT` | Playback frequency in Hz |
| `--loop` | Loop playback until the viewer closes |
| `--program {waypoints,sine}` | Built-in trajectory program when `--trajectries` is omitted |
| `--segT FLOAT` | Segment duration for waypoint programs |
| `--slow FLOAT` | Slow-motion playback factor |
| `--cameraDistance FLOAT` | Camera distance for `meshcat`, `matplotlib`, or `pyrender` |
| `--cameraAzimuth FLOAT` | Camera azimuth in degrees for `meshcat`, `matplotlib`, or `pyrender` |
| `--cameraElevation FLOAT` | Camera elevation in degrees for `meshcat`, `matplotlib`, or `pyrender` |
| `--cameraLookat X Y Z` | Camera look-at point for `meshcat`, `matplotlib`, or `pyrender` |
| `--record [PATH]` | Save backend output. Matplotlib and Pyrender write PNG or MP4, and MeshCat writes standalone HTML |
| `--recordFps FLOAT` | Override recording FPS for Matplotlib or Pyrender video export, or MeshCat HTML animation |
| `--recordSize W H` | Override video output resolution where supported |

The built-in renderers and the optional kinematics backend all read the same
URDF supplied via `--model`. MeshCat displays URDF `<visual>` geometry through
Pinocchio.

## Recording

```bash
uv run ei-vo-play \
  --renderer meshcat \
  --model examples/models/three_dof_arm.urdf \
  --trajectries examples/trajectories/three_dof_arm_waypoints.csv \
  --record recordings/scene.html

uv run ei-vo-play \
  --renderer matplotlib \
  --model examples/models/three_dof_arm.urdf \
  --trajectries examples/trajectories/three_dof_arm_waypoints.csv \
  --record recordings/link_frame.png

uv run ei-vo-play \
  --renderer pyrender \
  --model examples/models/three_dof_arm.urdf \
  --trajectries examples/trajectories/three_dof_arm_waypoints.csv \
  --record recordings/pyrender.mp4
```

Passing `--record` without a filename auto-generates a backend-specific output
name under `recordings/`. MeshCat writes a standalone `.html` snapshot of the
animated scene. Pyrender is offscreen-only and therefore always requires
`--record`; on headless Linux hosts it may also need `PYOPENGL_PLATFORM=egl`
or `PYOPENGL_PLATFORM=osmesa`. MP4 export requires `ffmpeg`.

## Trajectory files

Input files loaded via `--trajectries` must have shape `(T, DOF)`. `DOF` must
match the actuated arm joints detected in the URDF model. CSV, NPY, and JSON
are supported.

## Tests

```bash
uv run pytest
```
