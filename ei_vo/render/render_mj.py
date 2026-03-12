"""MuJoCo playback utilities."""

from __future__ import annotations

import contextlib
import pathlib
import re
import shutil
import tempfile
import time
import urllib.parse
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Mapping

import mujoco as mj
import mujoco.viewer as viewer
import numpy as np

from ..core import RobotModel, Trajectory


@dataclass(slots=True)
class ArmJointMap:
    """Resolved arm joint metadata for a MuJoCo model."""

    joint_ids: tuple[int, ...]
    joint_names: tuple[str, ...]
    qpos_addresses: tuple[int, ...]
    limits: np.ndarray

    @property
    def dof(self) -> int:
        return len(self.qpos_addresses)


@dataclass(slots=True)
class CameraSettings:
    """Viewer camera configuration."""

    distance: float | None = None
    azimuth: float | None = None
    elevation: float | None = None
    lookat: np.ndarray | tuple[float, float, float] | None = None

    def __post_init__(self) -> None:
        if self.lookat is not None:
            lookat = np.asarray(self.lookat, dtype=float)
            if lookat.shape != (3,):
                raise ValueError(f"lookat must have shape (3,). Got {lookat.shape}.")
            self.lookat = lookat

    @classmethod
    def from_camera(cls, camera: object) -> "CameraSettings":
        return cls(
            distance=getattr(camera, "distance", None),
            azimuth=getattr(camera, "azimuth", None),
            elevation=getattr(camera, "elevation", None),
            lookat=np.asarray(getattr(camera, "lookat", np.zeros(3)), dtype=float).copy(),
        )


@dataclass(slots=True)
class PlaybackConfig:
    """Playback timing behaviour."""

    hz: float = 240.0
    slow: float = 1.0
    loop: bool = False

    def __post_init__(self) -> None:
        if self.hz <= 0:
            raise ValueError(f"hz must be positive. Got {self.hz}.")
        if self.slow <= 0:
            raise ValueError(f"slow must be positive. Got {self.slow}.")

    @property
    def step_dt(self) -> float:
        return self.slow / self.hz


@dataclass(slots=True)
class RecordingConfig:
    """Recording settings for offscreen rendering."""

    path: str | pathlib.Path
    fps: float | None = None
    size: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        self.path = pathlib.Path(self.path)
        if self.fps is not None and self.fps <= 0:
            raise ValueError(f"fps must be positive. Got {self.fps}.")
        if self.size is not None:
            width, height = self.size
            if width <= 0 or height <= 0:
                raise ValueError(f"size must contain positive integers. Got {self.size}.")
            self.size = (int(width), int(height))


def _joint_sort_key(name: str) -> int:
    match = re.search(r"(\d+)$", name) or re.search(r"joint[_-]?(\d+)", name)
    return int(match.group(1)) if match else 999


def detect_arm_joints(model: mj.MjModel, expected_dof: int | None = None) -> ArmJointMap:
    """Collect arm hinge joints while skipping grippers and fingers."""

    joint_ids: list[int] = []
    joint_names: list[str] = []
    qpos_addresses: list[int] = []

    for joint_id in range(model.njnt):
        if model.jnt_type[joint_id] != mj.mjtJoint.mjJNT_HINGE:
            continue
        joint_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, joint_id) or ""
        lowered = joint_name.lower()
        if "finger" in lowered or "gripper" in lowered:
            continue
        joint_ids.append(joint_id)
        joint_names.append(joint_name)
        qpos_addresses.append(int(model.jnt_qposadr[joint_id]))

    if not joint_ids:
        raise RuntimeError("No arm hinge joints found in model.")

    order = np.argsort([_joint_sort_key(name) for name in joint_names])
    joint_ids = [joint_ids[index] for index in order]
    joint_names = [joint_names[index] for index in order]
    qpos_addresses = [qpos_addresses[index] for index in order]

    if expected_dof is not None:
        if len(joint_ids) < expected_dof:
            raise RuntimeError(
                f"Model provides {len(joint_ids)} arm joints, but {expected_dof} were requested."
            )
        joint_ids = joint_ids[:expected_dof]
        joint_names = joint_names[:expected_dof]
        qpos_addresses = qpos_addresses[:expected_dof]

    limits = np.asarray([model.jnt_range[joint_id] for joint_id in joint_ids], dtype=float)
    return ArmJointMap(
        joint_ids=tuple(joint_ids),
        joint_names=tuple(joint_names),
        qpos_addresses=tuple(qpos_addresses),
        limits=limits,
    )


def detect_arm_joint_qaddr(model: mj.MjModel, expected_dof: int | None = None) -> list[int]:
    """Compatibility helper returning only qpos addresses."""

    return list(detect_arm_joints(model, expected_dof=expected_dof).qpos_addresses)


def _resolve_urdf_mesh_source(model_dir: pathlib.Path, filename: str) -> pathlib.Path | None:
    parsed = urllib.parse.urlparse(filename)
    if parsed.scheme in {"", "file"}:
        raw_path = urllib.parse.unquote(parsed.path if parsed.scheme == "file" else filename)
        candidate = pathlib.Path(raw_path)
        if not candidate.is_absolute():
            candidate = model_dir / candidate
        return candidate if candidate.is_file() else None

    if parsed.scheme == "package":
        tail = pathlib.Path(urllib.parse.unquote(parsed.path.lstrip("/")))
        package_name = urllib.parse.unquote(parsed.netloc)
        search_roots = (model_dir, *model_dir.parents)
        for root in search_roots:
            for candidate in (root / tail, root / package_name / tail):
                if candidate.is_file():
                    return candidate
        return None

    return None


@contextlib.contextmanager
def _prepared_mujoco_model_path(model_path: str | pathlib.Path):
    path = pathlib.Path(model_path)
    if path.suffix.lower() != ".urdf":
        yield path
        return

    tree = ET.parse(path)
    root = tree.getroot()
    staged_assets: list[tuple[pathlib.Path, str]] = []

    # MuJoCo's URDF importer flattens mesh filenames to their basenames when
    # resolving mesh assets, so nested relative paths need to be staged.
    for index, mesh in enumerate(root.findall(".//mesh[@filename]")):
        filename = mesh.get("filename")
        if filename is None:
            continue
        source = _resolve_urdf_mesh_source(path.parent, filename)
        if source is None:
            continue
        staged_name = f"asset_{index:04d}{source.suffix}"
        mesh.set("filename", staged_name)
        staged_assets.append((source, staged_name))

    if not staged_assets:
        yield path
        return

    with tempfile.TemporaryDirectory(prefix="ei_vo_mj_urdf_") as tmp_dir:
        staged_dir = pathlib.Path(tmp_dir)
        staged_path = staged_dir / path.name
        tree.write(staged_path, encoding="utf-8", xml_declaration=True)
        for source, staged_name in staged_assets:
            shutil.copy2(source, staged_dir / staged_name)
        yield staged_path


def _load_mujoco_model(model_path: str | pathlib.Path) -> mj.MjModel:
    with _prepared_mujoco_model_path(model_path) as prepared_path:
        return mj.MjModel.from_xml_path(prepared_path.as_posix())


def load_robot_model(model_path: str | pathlib.Path, expected_dof: int | None = None) -> RobotModel:
    """Load a MuJoCo model and expose only arm-joint metadata."""

    path = pathlib.Path(model_path)
    model = _load_mujoco_model(path)
    arm_joints = detect_arm_joints(model, expected_dof=expected_dof)
    return RobotModel(
        name=path.stem,
        joint_names=arm_joints.joint_names,
        limits=arm_joints.limits,
    )


def _clip_positions_to_limits(q: np.ndarray, limits: np.ndarray) -> np.ndarray:
    positions = np.asarray(q, dtype=float).copy()
    if positions.ndim != 2:
        raise ValueError(f"Trajectory positions must be 2D. Got {positions.shape}.")
    if positions.shape[1] != limits.shape[0]:
        raise ValueError(
            f"Trajectory dof ({positions.shape[1]}) does not match detected joints ({limits.shape[0]})."
        )

    for index, (lower, upper) in enumerate(limits):
        if lower < upper:
            positions[:, index] = np.clip(positions[:, index], lower, upper)
    return positions


def clamp_to_limits(model: mj.MjModel, arm_qaddr: list[int], q: np.ndarray) -> np.ndarray:
    """Compatibility helper to clamp positions to model limits."""

    joint_limits = []
    for qpos_address in arm_qaddr:
        matches = np.flatnonzero(model.jnt_qposadr == qpos_address)
        if matches.size == 0:
            joint_limits.append(np.array([-np.inf, np.inf], dtype=float))
            continue
        joint_limits.append(np.asarray(model.jnt_range[int(matches[0])], dtype=float))
    return _clip_positions_to_limits(q, np.asarray(joint_limits, dtype=float))


def _apply_camera_settings(
    camera: object,
    settings: CameraSettings | Mapping[str, object] | None,
) -> None:
    if settings is None:
        return
    if isinstance(settings, Mapping):
        settings = CameraSettings(
            distance=settings.get("distance"),
            azimuth=settings.get("azimuth"),
            elevation=settings.get("elevation"),
            lookat=settings.get("lookat"),
        )

    if settings.distance is not None:
        camera.distance = float(settings.distance)
    if settings.azimuth is not None:
        camera.azimuth = float(settings.azimuth)
    if settings.elevation is not None:
        camera.elevation = float(settings.elevation)
    if settings.lookat is not None:
        camera.lookat[:] = settings.lookat


def _configure_default_camera(model: mj.MjModel, active_viewer: object) -> None:
    if hasattr(mj, "mjv_defaultFreeCamera"):
        mj.mjv_defaultFreeCamera(model, active_viewer.cam)
        return

    mj.mjv_defaultCamera(active_viewer.cam)
    if hasattr(model, "stat"):
        center = getattr(model.stat, "center", None)
        if center is not None:
            active_viewer.cam.lookat[:] = center
        extent = getattr(model.stat, "extent", None)
        if extent:
            active_viewer.cam.distance = extent


def _init_recording(
    model: mj.MjModel,
    dt: float,
    record_path: str | pathlib.Path,
    record_fps: float | None,
    record_size: tuple[int, int] | None,
):
    """Initialise offscreen rendering resources."""

    try:
        import imageio.v2 as imageio
    except ImportError as exc:
        raise RuntimeError("Recording requires the optional dependency 'imageio'.") from exc

    recording = RecordingConfig(path=record_path, fps=record_fps, size=record_size)
    path = recording.path
    if path.suffix == "":
        path = path.with_suffix(".mp4")
    path.parent.mkdir(parents=True, exist_ok=True)

    width, height = recording.size if recording.size is not None else (1280, 720)

    vis = getattr(model, "vis", None)
    global_vis = getattr(vis, "global_", None) if vis is not None else None
    if global_vis is not None and hasattr(global_vis, "offwidth") and hasattr(global_vis, "offheight"):
        global_vis.offwidth = max(int(global_vis.offwidth), int(width))
        global_vis.offheight = max(int(global_vis.offheight), int(height))

    renderer = mj.Renderer(model, height=height, width=width)
    camera = mj.MjvCamera()
    mj.mjv_defaultCamera(camera)
    fps = recording.fps if recording.fps is not None else (1.0 / max(dt, 1e-9))
    try:
        writer = imageio.get_writer(path.as_posix(), fps=fps)
    except ValueError as exc:
        raise RuntimeError(
            "MP4 recording requires an imageio video backend. "
            "Install 'imageio[ffmpeg]' (recommended) or 'imageio[pyav]'."
        ) from exc
    return renderer, camera, writer


def play_trajectory(
    model_path: str | pathlib.Path,
    trajectory: Trajectory | np.ndarray | list[list[float]] | list[float],
    *,
    playback: PlaybackConfig | None = None,
    camera: CameraSettings | Mapping[str, object] | None = None,
    recording: RecordingConfig | None = None,
) -> None:
    """Play a trajectory on top of a MuJoCo model."""

    path = pathlib.Path(model_path)
    model = _load_mujoco_model(path)
    data = mj.MjData(model)
    traj = Trajectory.coerce(trajectory)
    arm_joints = detect_arm_joints(model, expected_dof=traj.dof)
    playback = playback or PlaybackConfig()
    positions = _clip_positions_to_limits(traj.q, arm_joints.limits)

    record_renderer = None
    record_camera = None
    record_writer = None
    if recording is not None:
        record_renderer, record_camera, record_writer = _init_recording(
            model,
            playback.step_dt,
            recording.path,
            recording.fps,
            recording.size,
        )

    with contextlib.ExitStack() as stack:
        active_viewer = stack.enter_context(viewer.launch_passive(model, data))
        if record_writer is not None:
            stack.callback(record_writer.close)
        if record_renderer is not None:
            stack.callback(record_renderer.close)

        if camera is None:
            _configure_default_camera(model, active_viewer)
        else:
            _apply_camera_settings(active_viewer.cam, camera)

        if record_camera is not None:
            _apply_camera_settings(record_camera, CameraSettings.from_camera(active_viewer.cam))

        while active_viewer.is_running():
            for row in positions:
                for qpos_address, value in zip(arm_joints.qpos_addresses, row):
                    data.qpos[qpos_address] = float(value)
                mj.mj_forward(model, data)

                if record_renderer is not None and record_camera is not None and record_writer is not None:
                    _apply_camera_settings(record_camera, CameraSettings.from_camera(active_viewer.cam))
                    record_renderer.update_scene(data, camera=record_camera)
                    frame = record_renderer.render()
                    if frame.dtype != np.uint8:
                        frame = np.clip(frame * 255.0, 0.0, 255.0).astype(np.uint8)
                    record_writer.append_data(frame)

                active_viewer.sync()
                time.sleep(playback.step_dt)

            if not playback.loop:
                break

        while active_viewer.is_running():
            active_viewer.sync()
            time.sleep(0.01)


def play(
    model_path: str | pathlib.Path,
    traj: Trajectory | np.ndarray | list[list[float]] | list[float],
    slow: float = 1.0,
    hz: float = 240.0,
    camera: CameraSettings | Mapping[str, object] | None = None,
    loop: bool = False,
    record_path: str | pathlib.Path | None = None,
    record_fps: float | None = None,
    record_size: tuple[int, int] | None = None,
    *,
    kinematics_backend: str | None = None,
    kinematics_model_path: str | pathlib.Path | None = None,
    base_link: str | None = None,
    end_link: str | None = None,
) -> None:
    """Compatibility wrapper around :func:`play_trajectory`."""

    del kinematics_backend, kinematics_model_path, base_link, end_link

    recording = None
    if record_path is not None:
        recording = RecordingConfig(path=record_path, fps=record_fps, size=record_size)
    play_trajectory(
        model_path,
        traj,
        playback=PlaybackConfig(hz=hz, slow=slow, loop=loop),
        camera=camera,
        recording=recording,
    )


__all__ = [
    "ArmJointMap",
    "CameraSettings",
    "PlaybackConfig",
    "RecordingConfig",
    "_init_recording",
    "clamp_to_limits",
    "detect_arm_joint_qaddr",
    "detect_arm_joints",
    "load_robot_model",
    "play",
    "play_trajectory",
]
