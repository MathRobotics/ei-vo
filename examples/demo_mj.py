#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import inspect
import json
import math
import os
import pathlib
import sys
import time
import types
import warnings

import numpy as np
import mujoco as mj

from ei_vo import play
from ei_vo.render import render_mj

# ---------------------------
# ユーティリティ
# ---------------------------
def load_angles(path: str, deg: bool) -> np.ndarray:
    p = pathlib.Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    if p.suffix.lower() == ".csv":
        arr = np.loadtxt(p, delimiter=",", dtype=float)
    elif p.suffix.lower() == ".npy":
        arr = np.load(p)
    elif p.suffix.lower() == ".json":
        with open(p, "r") as f:
            data = json.load(f)
        arr = np.array(data, dtype=float)
    else:
        raise ValueError(f"Unsupported file extension: {p.suffix}")
    if arr.ndim != 2:
        raise ValueError(f"angles must be a 2D array. Got {arr.shape}")
    if deg:
        arr = np.deg2rad(arr)
    return arr

def quintic(q0: np.ndarray, q1: np.ndarray, T: float, dt: float) -> np.ndarray:
    """位置・速度・加速度=0で接続する5次多項式（スカラー係数）"""
    t = np.arange(0.0, T + 1e-12, dt)
    s = t / max(T, 1e-9)
    a = 10*s**3 - 15*s**4 + 6*s**5
    return q0[None, :] + (q1 - q0)[None, :] * a[:, None]

# ---------------------------
# デモ軌道生成（角度ファイルが無いとき）
# ---------------------------
def demo_waypoints(dof: int) -> np.ndarray:
    """安全めな代表姿勢を ``dof`` 関節分返す（行=姿勢、列=dof）"""

    base = np.linspace(-0.6, 0.6, dof, dtype=float)
    phase = np.linspace(0.0, math.pi, dof, dtype=float)

    offsets = [
        np.zeros(dof, dtype=float),
        0.35 * np.sin(phase),
        -0.25 * np.sin(phase + math.pi / 4.0),
        0.30 * np.sin(phase + math.pi / 2.0),
        np.zeros(dof, dtype=float),
    ]

    poses = [base + off for off in offsets]
    poses[-1] = poses[0].copy()  # 戻る
    return np.vstack(poses)

def build_demo_trajectory(q_wp: np.ndarray, seg_T: float, hz: float) -> np.ndarray:
    """ウェイポイント列を各区間 quintic で接続して結合した時系列を作る"""
    dt = 1.0 / max(hz, 1e-6)
    chunks = []
    for i in range(len(q_wp)-1):
        chunks.append(quintic(q_wp[i], q_wp[i+1], seg_T, dt)[:-1])  # 重複フレームを避けて末尾除外
    chunks.append(q_wp[-1][None, :])
    return np.vstack(chunks)

def build_sine_demo(dof: int, T_sec: float, hz: float) -> np.ndarray:
    """簡単なサイン波デモ（安全域の小振幅）。行=T*hz, 列=dof"""
    dt = 1.0 / max(hz, 1e-6)
    t = np.arange(0.0, T_sec + 1e-12, dt)
    T = t.shape[0]
    q = np.zeros((T, dof), dtype=float)
    base = np.linspace(-0.6, 0.6, dof, dtype=float)
    amp = np.linspace(0.15, 0.30, dof, dtype=float)
    freq = np.linspace(0.20, 0.35, dof, dtype=float)  # Hz
    phase = np.linspace(0.0, math.pi, dof, dtype=float)
    for i in range(dof):
        q[:, i] = base[i] + amp[i] * np.sin(2 * math.pi * freq[i] * t + phase[i])
    return q

# ---------------------------
# メイン
# ---------------------------
def _resolve_record_destination(record_arg):
    """Resolve the desired recording path.

    Returns a tuple of ``(path_or_none, auto_dir)`` where ``auto_dir`` is the
    directory that was auto-created when the caller did not supply an explicit
    filename. ``auto_dir`` is ``None`` when the caller provided a concrete
    destination.
    """

    if record_arg is None:
        return None, None

    record_value = os.fspath(record_arg)
    record_value = record_value.strip()

    if record_value == "":
        base_dir = pathlib.Path.cwd() / "recordings"
    else:
        candidate = pathlib.Path(record_value)
        if record_value.endswith(os.sep):
            base_dir = candidate
        elif candidate.exists() and candidate.is_dir():
            base_dir = candidate
        else:
            return candidate.as_posix(), None

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"demo_{timestamp}.mp4"
    record_path = (base_dir / filename).as_posix()
    return record_path, base_dir.as_posix()


def _prepare_play_invocation(args, traj_obj):
    """Build argument lists for ``ei.play`` depending on its signature."""

    sig = inspect.signature(play)
    params = sig.parameters

    call_args = [args.model]
    call_kwargs = {}

    if "traj" in params:
        call_kwargs["traj"] = traj_obj
    else:
        call_args.append(traj_obj)

    if "slow" in params:
        call_kwargs["slow"] = args.slow
    if "hz" in params:
        call_kwargs["hz"] = args.hz
    if "loop" in params:
        call_kwargs["loop"] = args.loop

    record_requested = args.record is not None
    record_supported = "record_path" in params
    fps_supported = "record_fps" in params
    size_supported = "record_size" in params

    if record_requested and record_supported:
        call_kwargs["record_path"] = args.record
        if fps_supported and args.recordFps is not None:
            call_kwargs["record_fps"] = args.recordFps
        if size_supported and args.recordSize:
            call_kwargs["record_size"] = tuple(args.recordSize)
        if not fps_supported and args.recordFps is not None:
            warnings.warn(
                "recordFps specified but ei.play() does not accept a record_fps argument; ignoring.",
                RuntimeWarning,
            )
        if not size_supported and args.recordSize:
            warnings.warn(
                "recordSize specified but ei.play() does not accept a record_size argument; ignoring.",
                RuntimeWarning,
            )
    else:
        if record_requested:
            warnings.warn(
                "Recording was requested but the installed ei.play() does not accept a 'record_path' argument.",
                RuntimeWarning,
            )
        if args.recordFps is not None:
            warnings.warn(
                "recordFps specified without recording support; ignoring.",
                RuntimeWarning,
            )
        if args.recordSize:
            warnings.warn(
                "recordSize specified without recording support; ignoring.",
                RuntimeWarning,
            )

    return call_args, call_kwargs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Panda の MJCF パス (panda.xml)")
    ap.add_argument("--angles", default=None, help="角度ファイル CSV/NPY/JSON, shape=(T, DOF)（省略可）")
    ap.add_argument("--deg", action="store_true", help="角度ファイルが度[deg]の場合に指定")
    ap.add_argument("--hz", type=float, default=240.0, help="再生周波数 [Hz]（デモ/ファイル共通）")
    ap.add_argument("--loop", action="store_true", help="終端でループ再生")
    ap.add_argument("--demo", choices=["wp", "sine"], default="wp", help="角度ファイルが無い時のデモ種別: wp(ウェイポイント) / sine(サイン波)")
    ap.add_argument("--segT", type=float, default=1.5, help="デモ=wp の各区間時間 [s]")
    ap.add_argument("--slow", type=float, default=1.0, help="実時間のスロー倍率（>1でゆっくり）")
    ap.add_argument(
        "--record",
        nargs="?",
        const="",
        default=None,
        help="録画動画の保存パス (例: output.mp4)。省略値またはディレクトリのみ指定時は recordings/ 以下に保存",
    )
    ap.add_argument("--recordFps", type=float, default=None, help="録画フレームレート [fps]（省略時: 再生fpsと同じ）")
    ap.add_argument("--recordSize", type=int, nargs=2, metavar=("W", "H"), default=None,
                    help="録画映像の幅[px]と高さ[px]（省略時: 1280x720）")

    args = ap.parse_args()

    if not os.path.isfile(args.model):
        raise FileNotFoundError(args.model)

    mj_model = mj.MjModel.from_xml_path(args.model)
    model_dof = len(render_mj.detect_arm_joint_qaddr(mj_model))
    if model_dof == 0:
        raise RuntimeError("モデルから腕関節を特定できませんでした")

    record_path, auto_dir = _resolve_record_destination(args.record)
    args.record = record_path

    if auto_dir is not None and record_path is not None:
        print(f"[demo_mj] --record でファイル名が指定されなかったため {record_path} に保存します")

    # 軌道用意
    if args.angles is None:
        if args.demo == "wp":
            q_wp = demo_waypoints(model_dof)
            q = build_demo_trajectory(q_wp, seg_T=args.segT, hz=args.hz)
        else:
            q = build_sine_demo(model_dof, T_sec=10.0, hz=args.hz)
    else:
        q = load_angles(args.angles, deg=args.deg)
        if q.shape[1] != model_dof:
            raise ValueError(
                f"角度ファイルの列数({q.shape[1]})がモデルの自由度({model_dof})と一致しません"
            )

    traj_obj = types.SimpleNamespace(q=q)
    call_args, call_kwargs = _prepare_play_invocation(args, traj_obj)
    play(*call_args, **call_kwargs)

if __name__ == "__main__":
    main()
