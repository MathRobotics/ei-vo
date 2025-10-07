from typing import Optional, Tuple, Union

from ..core.core import Trajectory
import mujoco as mj, mujoco.viewer as viewer
import numpy as np, time, pathlib
import contextlib

def detect_arm_joint_qaddr(m: mj.MjModel):
    """finger/gripper を除外し、腕7ヒンジの qpos index と名前を抽出"""
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

    # 名前末尾の番号や "jointX" を拾って 1..7 を優先ソート
    def key(nm: str):
        import re
        mnum = re.search(r"(\d+)$", nm) or re.search(r"joint[_-]?(\d+)", nm)
        return int(mnum.group(1)) if mnum else 999

    order = np.argsort([key(n) for n in names])
    qaddrs = [qaddrs[i] for i in order][:7]
    names  = [names[i]  for i in order][:7]

    if len(qaddrs) != 7:
        raise RuntimeError(f"7 arm joints not found. found={len(qaddrs)}, names={names}")
    return qaddrs

def clamp_to_limits(m: mj.MjModel, arm_qaddr: list[int], q: np.ndarray) -> np.ndarray:
    """モデルの関節範囲 m.jnt_range に基づいて角度をクリップ（必要な場合のみ）"""
    q = np.array(q, dtype=float)
    for i, adr in enumerate(arm_qaddr):
        # jnt_range は joint index 基準なので、adr から逆引きは不要（範囲が -inf の場合もある）
        # qpos index adr に対応する joint id を取得
        j_id = np.where(m.jnt_qposadr == adr)[0]
        if len(j_id) == 0:
            continue
        j = int(j_id[0])
        low, high = m.jnt_range[j]
        if low < high:  # 有効範囲が設定されている場合のみ
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
    arm_qaddr = detect_arm_joint_qaddr(m)

    dt = (1.0 / max(hz, 1e-6)) * max(slow, 1e-6)

    q = clamp_to_limits(m, arm_qaddr, traj.q)

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
                mj.mj_forward(m, d)  # 物理なしで姿勢だけ更新
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

        # 終了までウィンドウを残す（任意）
        while v.is_running():
            v.sync()
            time.sleep(0.01)
