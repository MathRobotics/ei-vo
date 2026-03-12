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
ei-vo-play --model examples/models/three_dof_arm.urdf --program waypoints
ei-vo-play --renderer meshcat --model examples/models/three_dof_arm.urdf --program waypoints
ei-vo-play --renderer matplotlib --model examples/models/three_dof_arm.urdf
ei-vo-play --renderer mujoco --model examples/models/three_dof_arm.urdf --backend pinocchio --base-link base --end-link ee
```

The first command uses the default `meshcat` renderer, Pinocchio for URDF
visual playback, and the default `literobo` backend.

On macOS, `ei-vo-play --renderer mujoco ...` and `ei-vo-demo --renderer mujoco
...` relaunch themselves via `mjpython`.

Compatibility wrapper:

```bash
ei-vo-demo --renderer mujoco --model examples/models/three_dof_arm.urdf --demo wp
python examples/demo_mj.py --renderer mujoco --model examples/models/three_dof_arm.urdf --demo wp
```

## Running the Python example

```bash
python examples/switch_renderer.py
```

Edit `MODEL`, `RENDERER`, and `BACKEND` to switch the playback setup. The file
defaults to `meshcat` plus `literobo`. If you change `RENDERER` to `"mujoco"`
on macOS, the script relaunches itself via `mjpython`.

## Options

| Option | Description |
| --- | --- |
| `--renderer {matplotlib,meshcat,mujoco}` | Select the renderer backend |
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
| `--record [PATH]` | Save backend output. MuJoCo writes MP4, MeshCat writes HTML, Matplotlib writes PNG |
| `--recordFps FLOAT` | Override recording FPS for MuJoCo |
| `--recordSize W H` | Override output resolution |

The built-in renderers and the optional kinematics backend all read the same
URDF supplied via `--model`. MeshCat displays URDF `<visual>` geometry through
Pinocchio. XML/MJCF input is not supported in the CLI.

## Recording

```bash
ei-vo-play \
  --renderer mujoco \
  --model examples/models/three_dof_arm.urdf \
  --record recordings/playback.mp4 \
  --recordFps 60

ei-vo-play \
  --renderer meshcat \
  --model examples/models/three_dof_arm.urdf \
  --trajectries examples/trajectories/three_dof_arm_waypoints.csv \
  --record recordings/scene.html

ei-vo-play \
  --renderer matplotlib \
  --model examples/models/three_dof_arm.urdf \
  --trajectries examples/trajectories/three_dof_arm_waypoints.csv \
  --record recordings/link_frame.png
```

Passing `--record` without a filename auto-generates a backend-specific output
name under `recordings/`.

## Trajectory files

Input files loaded via `--trajectries` must have shape `(T, DOF)`. For MuJoCo,
`DOF` must match the detected arm joints in the model. CSV, NPY, and JSON are
supported.

## Tests

```bash
pytest
```
