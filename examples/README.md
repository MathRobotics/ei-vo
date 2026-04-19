# Examples

`examples/demo_mj.py` is only a thin compatibility wrapper around the generic
package CLI. The CLI can target multiple renderers without mixing their
dependencies.

The single Python example is:

- `examples/switch_renderer.py`: switch the renderer and backend by editing variables

Bundled assets:

- `examples/models/simple_model.xml`
- `examples/models/three_dof_arm.urdf`
- `examples/trajectories/three_dof_arm_waypoints.csv`

## Running playback

```bash
uv run ei-vo-play --model examples/models/three_dof_arm.urdf --program waypoints
uv run ei-vo-play --renderer meshcat --model examples/models/three_dof_arm.urdf --program waypoints
uv run ei-vo-play --renderer matplotlib --model examples/models/three_dof_arm.urdf
uv run ei-vo-play --renderer mujoco --model examples/models/three_dof_arm.urdf --backend pinocchio --base-link base --end-link ee
uv run ei-vo-play --renderer blender --model examples/models/three_dof_arm.urdf --program waypoints --record recordings/blender.mp4
```

The first command uses the default `meshcat` renderer, Pinocchio for URDF
visual playback, and the default `pinocchio` backend.

On macOS, `ei-vo-play --renderer mujoco ...` and `ei-vo-demo --renderer mujoco
...` relaunch themselves via `mjpython`.

Compatibility wrapper:

```bash
uv run ei-vo-demo --renderer mujoco --model examples/models/three_dof_arm.urdf --demo wp
uv run python examples/demo_mj.py --renderer mujoco --model examples/models/three_dof_arm.urdf --demo wp
```

## Running the Python example

```bash
uv run python examples/switch_renderer.py
```

Edit `MODEL`, `RENDERER`, and `BACKEND` to switch the playback setup. The file
defaults to `meshcat` plus `pinocchio`. If you change `RENDERER` to `"mujoco"`
on macOS, the script relaunches itself via `mjpython`.

## Options

| Option | Description |
| --- | --- |
| `--renderer {blender,matplotlib,meshcat,mujoco}` | Select the renderer backend |
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
| `--cameraDistance FLOAT` | Camera distance for `mujoco`, `meshcat`, or `matplotlib` |
| `--cameraAzimuth FLOAT` | Camera azimuth in degrees for `mujoco`, `meshcat`, or `matplotlib` |
| `--cameraElevation FLOAT` | Camera elevation in degrees for `mujoco`, `meshcat`, or `matplotlib` |
| `--cameraLookat X Y Z` | Camera look-at point for `mujoco`, `meshcat`, or `matplotlib` |
| `--record [PATH]` | Save backend output. Blender and MuJoCo write MP4, Matplotlib writes PNG or MP4, and MeshCat writes standalone HTML |
| `--recordFps FLOAT` | Override recording FPS for Blender, MuJoCo, Matplotlib video export, or MeshCat HTML animation |
| `--recordSize W H` | Override video output resolution where supported |
| `--blenderEngine {workbench,eevee,cycles}` | Override the Blender render engine |
| `--blenderSamples INT` | Override the Blender sample count |

The built-in renderers and the optional kinematics backend all read the same
URDF supplied via `--model`. MeshCat displays URDF `<visual>` geometry through
Pinocchio. XML/MJCF input is not supported in the CLI.

## Recording

```bash
uv run ei-vo-play \
  --renderer mujoco \
  --model examples/models/three_dof_arm.urdf \
  --record recordings/playback.mp4 \
  --recordFps 60

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
  --renderer blender \
  --model examples/models/three_dof_arm.urdf \
  --program waypoints \
  --record recordings/blender.mp4
```

Passing `--record` without a filename auto-generates a backend-specific output
name under `recordings/`. MeshCat writes a standalone `.html` snapshot of the
animated scene. Blender prefers a local Blender installation and the
`blender` executable on your `PATH` or `EI_VO_BLENDER`; if that fails, `ei-vo`
can also render through the optional `bpy` module installed into the current
environment with `uv add bpy`. Repeated Blender renders reuse a cached `.blend`
scene under `$TMPDIR/ei_vo_blender_cache`. MP4 export requires `ffmpeg`.

## Trajectory files

Input files loaded via `--trajectries` must have shape `(T, DOF)`. For MuJoCo,
`DOF` must match the detected arm joints in the model. CSV, NPY, and JSON are
supported.

## Tests

```bash
uv run pytest
```
