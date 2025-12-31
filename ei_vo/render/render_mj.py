from typing import Optional, Tuple, Union

from ..core.core import Trajectory
import mujoco as mj, mujoco.viewer as viewer
import numpy as np, time, pathlib
import contextlib

def detect_arm_joint_qaddr(m: mj.MjModel, expected_dof: Optional[int] = None):
    """Collect qpos indices and names for arm hinge joints, skipping fingers/grippers."""
    qaddrs, names = [], []
    for j_id in range(m.njnt):
        if m.jnt_type[j_id] != mj.mjtJoint.mjJNT_HINGE:
            continue
        nm = mj.mj_id2name(m, mj.mjtObj.mjOBJ_JOINT, j_id) or ""
        low = nm.lower()
        if "finger" in low or "gripper" in low:
            continue
        qaddrs.append(int(m.jnt_qposadr[j_id]))
        names.append(nm)

    # Prefer sorting by trailing numbers such as "joint5" so links stay in order.
    def key(nm: str):
        import re
        mnum = re.search(r"(\d+)$", nm) or re.search(r"joint[_-]?(\d+)", nm)
        return int(mnum.group(1)) if mnum else 999

    order = np.argsort([key(n) for n in names])
    qaddrs = [qaddrs[i] for i in order]
    names = [names[i] for i in order]

    if expected_dof is None:
        if len(qaddrs) == 0:
            raise RuntimeError("No arm hinge joints found in model.")
        return qaddrs

    if len(qaddrs) < expected_dof:
        raise RuntimeError(
            f"Model provides {len(qaddrs)} arm joints, but {expected_dof} were requested."
        )
    qaddrs = qaddrs[:expected_dof]
    return qaddrs

def clamp_to_limits(m: mj.MjModel, arm_qaddr: list[int], q: np.ndarray) -> np.ndarray:
    """Clip the trajectory to ``m.jnt_range`` when joint limits are finite."""
    q = np.array(q, dtype=float)
    if q.ndim != 2:
        raise ValueError("Trajectory q must be a 2D array of shape (T, dof)")
    if q.shape[1] != len(arm_qaddr):
        raise ValueError(
            f"Trajectory dof ({q.shape[1]}) does not match detected joints ({len(arm_qaddr)})"
        )
    for i, adr in enumerate(arm_qaddr):
        # ``jnt_range`` is indexed by joint id, not by qpos adr; look up the
        # matching joint for each qpos index. Ranges can be infinite.
        j_id = np.where(m.jnt_qposadr == adr)[0]
        if len(j_id) == 0:
            continue
        j = int(j_id[0])
        low, high = m.jnt_range[j]
        if low < high:  # Only clamp when finite bounds exist
            q[:, i] = np.clip(q[:, i], low, high)
    return q

def _init_recording(m: mj.MjModel, dt: float, record_path: Union[str, pathlib.Path],
                    record_fps: Optional[float], record_size: Optional[Tuple[int, int]]):
    """Initialise offscreen renderer and video writer for recording."""
    try:
        import imageio.v2 as imageio
    except ImportError as exc:
        raise RuntimeError(
            "Recording requires the optional dependency 'imageio'."
        ) from exc

    path = pathlib.Path(record_path)
    if path.suffix == "":
        path = path.with_suffix(".mp4")
    path.parent.mkdir(parents=True, exist_ok=True)

    if record_size is None:
        width, height = 1280, 720
    else:
        width, height = record_size
        if width <= 0 or height <= 0:
            raise ValueError("record_size must contain positive integers")

    # ``mj.Renderer`` requires the offscreen framebuffer size (specified by
    # ``offwidth``/``offheight`` in the MJCF) to be large enough for the
    # requested render size.  Many Panda models ship with the default
    # 640x480 buffer, which triggers a ``ValueError`` when users ask for HD
    # recordings.  Adjust the framebuffer dimensions on the loaded model so
    # that recording works without requiring manual MJCF edits.
    vis = getattr(m, "vis", None)
    global_vis = getattr(vis, "global_", None) if vis is not None else None
    if global_vis is not None and hasattr(global_vis, "offwidth") and hasattr(global_vis, "offheight"):
        global_vis.offwidth = max(int(global_vis.offwidth), int(width))
        global_vis.offheight = max(int(global_vis.offheight), int(height))

    renderer = mj.Renderer(m, height=height, width=width)
    camera = mj.MjvCamera()
    mj.mjv_defaultCamera(camera)

    fps = record_fps if record_fps and record_fps > 0 else (1.0 / max(dt, 1e-9))
    writer = imageio.get_writer(path.as_posix(), fps=fps)
    return renderer, camera, writer


def play(model_path: str, traj: Trajectory, slow=1.0, hz=240.0, camera=None, loop=False,
         record_path: Optional[str] = None, record_fps: Optional[float] = None,
         record_size: Optional[Tuple[int, int]] = None):
    m = mj.MjModel.from_xml_path(model_path)
    d = mj.MjData(m)
    q = np.array(traj.q, dtype=float)
    if q.ndim != 2:
        raise ValueError("traj.q must be a 2D array of shape (T, dof)")

    arm_qaddr = detect_arm_joint_qaddr(m, expected_dof=q.shape[1])

    dt = (1.0 / max(hz, 1e-6)) * max(slow, 1e-6)

    q = clamp_to_limits(m, arm_qaddr, q)

    record_renderer = None
    record_camera = None
    record_writer = None
    if record_path is not None:
        record_renderer, record_camera, record_writer = _init_recording(
            m, dt, record_path, record_fps, record_size
        )

    def _apply_camera_settings(cam_obj, settings):
        if settings is None:
            return
        for key, value in settings.items():
            if key == "lookat":
                cam_obj.lookat[:] = np.asarray(value, dtype=float)
            else:
                setattr(cam_obj, key, value)

    with contextlib.ExitStack() as stack:
        v = stack.enter_context(viewer.launch_passive(m, d))
        if record_writer is not None:
            stack.callback(record_writer.close)
        if record_renderer is not None:
            stack.callback(record_renderer.close)

        if camera is None:
            if hasattr(mj, "mjv_defaultFreeCamera"):
                mj.mjv_defaultFreeCamera(m, v.cam)
            else:
                mj.mjv_defaultCamera(v.cam)
                if hasattr(m, "stat"):
                    try:
                        v.cam.lookat[:] = getattr(m.stat, "center")
                    except Exception:
                        pass
                    distance = getattr(m.stat, "extent", None)
                    if distance:
                        v.cam.distance = distance
        else:
            _apply_camera_settings(v.cam, camera)

        if record_camera is not None:
            _apply_camera_settings(record_camera, {
                "distance": v.cam.distance,
                "azimuth": v.cam.azimuth,
                "elevation": v.cam.elevation,
                "lookat": v.cam.lookat,
            })

        def play_once(v):
            for i in range(q.shape[0]):
                for adr, qi in zip(arm_qaddr, q[i]):
                    d.qpos[adr] = float(qi)
                mj.mj_forward(m, d)  # Update pose without running dynamics
                if (record_renderer is not None and record_writer is not None
                        and record_camera is not None):
                    _apply_camera_settings(record_camera, {
                        "distance": v.cam.distance,
                        "azimuth": v.cam.azimuth,
                        "elevation": v.cam.elevation,
                        "lookat": v.cam.lookat,
                    })
                    record_renderer.update_scene(d, camera=record_camera)
                    frame = record_renderer.render()
                    if frame.dtype != np.uint8:
                        frame = np.clip(frame * 255, 0, 255).astype(np.uint8)
                    record_writer.append_data(frame)
                v.sync()
                time.sleep(dt)

        while v.is_running():
            play_once(v)
            if not loop:
                break

        # Keep the viewer window open until the user closes it (optional).
        while v.is_running():
            v.sync()
            time.sleep(0.01)
