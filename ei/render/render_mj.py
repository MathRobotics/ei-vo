from ..core.core import RobotModel, Trajectory
import mujoco as mj, mujoco.viewer as viewer
import numpy as np, time

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

def play(model_path: str, traj: Trajectory, slow=1.0, hz=240.0, camera=None, loop=False):
    m = mj.MjModel.from_xml_path(model_path)
    d = mj.MjData(m)
    arm_qaddr = detect_arm_joint_qaddr(m)

    dt = (1.0 / max(hz, 1e-6)) * max(slow, 1e-6)

    q = clamp_to_limits(m, arm_qaddr, traj.q)

    with viewer.launch_passive(m, d) as v:
        # カメラ
        v.cam.distance = 1.9
        v.cam.azimuth  = 110
        v.cam.elevation= -20

        def play_once(v):
            for i in range(q.shape[0]):
                for adr, qi in zip(arm_qaddr, q[i]):
                    d.qpos[adr] = float(qi)
                mj.mj_forward(m, d)  # 物理なしで姿勢だけ更新
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
