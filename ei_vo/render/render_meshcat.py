"""MeshCat-based playback backend."""

from __future__ import annotations

import os
import pathlib
import shutil
import subprocess
import tempfile
import time
from typing import Mapping

import numpy as np

from ..core import Trajectory
from .render_mj import CameraSettings, PlaybackConfig, detect_arm_joints


def _import_meshcat():
    try:
        import meshcat
    except ImportError as exc:
        raise RuntimeError(
            "The 'meshcat' renderer requires the optional dependency 'meshcat'."
        ) from exc
    return meshcat


def _import_mujoco():
    try:
        import mujoco as mj
    except ImportError as exc:
        raise RuntimeError(
            "The 'meshcat' renderer requires the optional dependency 'mujoco' for non-URDF models."
        ) from exc
    return mj


def _import_imageio():
    try:
        import imageio.v2 as imageio
    except ImportError as exc:
        raise RuntimeError(
            "MeshCat video recording requires the optional dependency 'imageio[ffmpeg]'."
        ) from exc
    return imageio


def _import_pinocchio_visualizer():
    try:
        import pinocchio as pin
        from pinocchio.visualize import MeshcatVisualizer
    except ImportError as exc:
        raise RuntimeError(
            "The 'meshcat' renderer requires the optional dependency 'pin' for URDF visual playback."
        ) from exc
    return pin, MeshcatVisualizer


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


def _clip_positions_to_pinocchio_limits(
    q: np.ndarray,
    lower_limits: np.ndarray,
    upper_limits: np.ndarray,
) -> np.ndarray:
    positions = np.asarray(q, dtype=float).copy()
    lower = np.asarray(lower_limits, dtype=float).reshape(-1)
    upper = np.asarray(upper_limits, dtype=float).reshape(-1)
    if positions.ndim != 2:
        raise ValueError(f"Trajectory positions must be 2D. Got {positions.shape}.")
    if positions.shape[1] != lower.shape[0] or positions.shape[1] != upper.shape[0]:
        raise ValueError(
            "Trajectory dof "
            f"({positions.shape[1]}) does not match Pinocchio model dof ({lower.shape[0]})."
        )
    for index, (lower_bound, upper_bound) in enumerate(zip(lower, upper)):
        if np.isfinite(lower_bound) and np.isfinite(upper_bound) and lower_bound < upper_bound:
            positions[:, index] = np.clip(positions[:, index], lower_bound, upper_bound)
    return positions


def _make_transform(
    translation: np.ndarray,
    rotation: np.ndarray | None = None,
    scale: np.ndarray | None = None,
) -> np.ndarray:
    transform = np.eye(4, dtype=float)
    if rotation is not None:
        transform[:3, :3] = np.asarray(rotation, dtype=float).reshape(3, 3)
    if scale is not None:
        transform[:3, :3] = transform[:3, :3] @ np.diag(np.asarray(scale, dtype=float))
    transform[:3, 3] = np.asarray(translation, dtype=float)
    return transform


def _rotation_x(theta: float) -> np.ndarray:
    cos_theta = float(np.cos(theta))
    sin_theta = float(np.sin(theta))
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cos_theta, -sin_theta],
            [0.0, sin_theta, cos_theta],
        ],
        dtype=float,
    )


_MESHCAT_CAMERA_ROOT_PATH = "/Cameras/default"
_MESHCAT_CAMERA_OBJECT_PATH = "/Cameras/default/rotated/<object>"
_MESHCAT_DEFAULT_CAMERA_POSITION_ROTATED = np.array([3.0, 1.0, 0.0], dtype=float)
_MESHCAT_ROTATED_TO_WORLD = _rotation_x(np.pi / 2.0)
_MESHCAT_WORLD_TO_ROTATED = _rotation_x(-np.pi / 2.0)


def _coerce_camera_settings(camera: object | Mapping[str, object] | None) -> CameraSettings | None:
    if camera is None:
        return None
    if isinstance(camera, CameraSettings):
        return camera
    if isinstance(camera, Mapping):
        return CameraSettings(
            distance=camera.get("distance"),
            azimuth=camera.get("azimuth"),
            elevation=camera.get("elevation"),
            lookat=camera.get("lookat"),
        )
    return CameraSettings.from_camera(camera)


def _camera_settings_from_world_offset(world_offset: np.ndarray) -> CameraSettings:
    offset = np.asarray(world_offset, dtype=float).reshape(3)
    distance = float(np.linalg.norm(offset))
    if distance <= 1e-12:
        return CameraSettings(distance=0.0, azimuth=0.0, elevation=0.0, lookat=np.zeros(3))
    azimuth = float(np.degrees(np.arctan2(-offset[1], -offset[0])))
    elevation = float(np.degrees(np.arctan2(-offset[2], np.hypot(offset[0], offset[1]))))
    return CameraSettings(distance=distance, azimuth=azimuth, elevation=elevation, lookat=np.zeros(3))


_MESHCAT_DEFAULT_CAMERA_SETTINGS = _camera_settings_from_world_offset(
    _MESHCAT_ROTATED_TO_WORLD @ _MESHCAT_DEFAULT_CAMERA_POSITION_ROTATED
)


def _resolve_meshcat_camera_settings(
    camera: object | Mapping[str, object] | None,
) -> CameraSettings | None:
    settings = _coerce_camera_settings(camera)
    if settings is None:
        return None
    defaults = _MESHCAT_DEFAULT_CAMERA_SETTINGS
    return CameraSettings(
        distance=defaults.distance if settings.distance is None else float(settings.distance),
        azimuth=defaults.azimuth if settings.azimuth is None else float(settings.azimuth),
        elevation=defaults.elevation if settings.elevation is None else float(settings.elevation),
        lookat=np.zeros(3, dtype=float) if settings.lookat is None else np.asarray(settings.lookat, dtype=float),
    )


def _camera_world_offset(settings: CameraSettings) -> np.ndarray:
    azimuth = float(np.deg2rad(settings.azimuth))
    elevation = float(np.deg2rad(settings.elevation))
    cos_elevation = float(np.cos(elevation))
    return np.array(
        [
            -float(settings.distance) * cos_elevation * float(np.cos(azimuth)),
            -float(settings.distance) * cos_elevation * float(np.sin(azimuth)),
            -float(settings.distance) * float(np.sin(elevation)),
        ],
        dtype=float,
    )


def _apply_camera_settings(
    visualizer,
    camera: object | Mapping[str, object] | None,
) -> None:
    settings = _resolve_meshcat_camera_settings(camera)
    if settings is None:
        return

    visualizer[_MESHCAT_CAMERA_ROOT_PATH].set_transform(
        _make_transform(np.asarray(settings.lookat, dtype=float))
    )
    rotated_offset = _MESHCAT_WORLD_TO_ROTATED @ _camera_world_offset(settings)
    rotated_offset[np.abs(rotated_offset) <= 1e-12] = 0.0
    visualizer[_MESHCAT_CAMERA_OBJECT_PATH].set_property(
        "position",
        [float(value) for value in rotated_offset],
    )


def _rgba_to_material(meshcat_geometry, rgba: np.ndarray):
    r, g, b, a = np.asarray(rgba, dtype=float)
    color = (int(np.clip(r, 0.0, 1.0) * 255) << 16) | (int(np.clip(g, 0.0, 1.0) * 255) << 8) | int(
        np.clip(b, 0.0, 1.0) * 255
    )
    transparent = bool(a < 0.999)
    for material_name in ("MeshPhongMaterial", "MeshLambertMaterial", "MeshBasicMaterial"):
        material_cls = getattr(meshcat_geometry, material_name, None)
        if material_cls is not None:
            return material_cls(color=color, opacity=float(a), transparent=transparent)
    return None


def _geom_rgba(model: mj.MjModel, geom_id: int) -> np.ndarray:
    if hasattr(model, "geom_rgba"):
        return np.asarray(model.geom_rgba[geom_id], dtype=float)
    return np.array([0.7, 0.7, 0.7, 1.0], dtype=float)


def _set_geom_object(
    visualizer,
    meshcat_geometry,
    mj,
    geom_type: int,
    geom_id: int,
    geom_size: np.ndarray,
    rgba: np.ndarray,
) -> bool:
    geom_node = visualizer[f"geoms/{geom_id}"]
    material = _rgba_to_material(meshcat_geometry, rgba)

    if geom_type == getattr(mj.mjtGeom, "mjGEOM_SPHERE", -1):
        geom_node["shape"].set_object(meshcat_geometry.Sphere(float(geom_size[0])), material)
        return True

    if geom_type == getattr(mj.mjtGeom, "mjGEOM_BOX", -1):
        geom_node["shape"].set_object(meshcat_geometry.Box((2.0 * geom_size[:3]).tolist()), material)
        return True

    if geom_type == getattr(mj.mjtGeom, "mjGEOM_CYLINDER", -1):
        geom_node["shape"].set_object(
            meshcat_geometry.Cylinder(2.0 * float(geom_size[1]), float(geom_size[0])),
            material,
        )
        geom_node["shape"].set_transform(_make_transform(np.zeros(3), _rotation_x(np.pi / 2.0)))
        return True

    if geom_type == getattr(mj.mjtGeom, "mjGEOM_CAPSULE", -1):
        geom_node["cylinder"].set_object(
            meshcat_geometry.Cylinder(2.0 * float(geom_size[1]), float(geom_size[0])),
            material,
        )
        geom_node["cylinder"].set_transform(_make_transform(np.zeros(3), _rotation_x(np.pi / 2.0)))
        for axis_sign, node_name in ((1.0, "cap_pos"), (-1.0, "cap_neg")):
            geom_node[node_name].set_object(meshcat_geometry.Sphere(float(geom_size[0])), material)
            geom_node[node_name].set_transform(
                _make_transform(np.array([0.0, 0.0, axis_sign * float(geom_size[1])]))
            )
        return True

    if geom_type == getattr(mj.mjtGeom, "mjGEOM_ELLIPSOID", -1):
        geom_node["shape"].set_object(meshcat_geometry.Sphere(1.0), material)
        geom_node["shape"].set_transform(_make_transform(np.zeros(3), scale=geom_size[:3]))
        return True

    return False


def _create_scene(visualizer, model: mj.MjModel) -> tuple[int, ...]:
    supported_geom_ids: list[int] = []
    mj = _import_mujoco()
    try:
        import meshcat.geometry as meshcat_geometry
    except ImportError as exc:
        raise RuntimeError(
            "The 'meshcat' renderer requires the optional dependency 'meshcat'."
        ) from exc

    for geom_id in range(getattr(model, "ngeom", 0)):
        geom_type = int(model.geom_type[geom_id])
        geom_size = np.asarray(model.geom_size[geom_id], dtype=float)
        rgba = _geom_rgba(model, geom_id)
        if _set_geom_object(visualizer, meshcat_geometry, mj, geom_type, geom_id, geom_size, rgba):
            supported_geom_ids.append(geom_id)
    return tuple(supported_geom_ids)


def _update_scene(visualizer, data: mj.MjData, geom_ids: tuple[int, ...]) -> None:
    for geom_id in geom_ids:
        rotation = np.asarray(data.geom_xmat[geom_id], dtype=float).reshape(3, 3)
        translation = np.asarray(data.geom_xpos[geom_id], dtype=float)
        visualizer[f"geoms/{geom_id}"].set_transform(_make_transform(translation, rotation))


def _save_html(visualizer, record_path: str | pathlib.Path) -> pathlib.Path:
    path = pathlib.Path(record_path)
    if path.suffix == "":
        path = path.with_suffix(".html")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not hasattr(visualizer, "static_html"):
        raise RuntimeError("The installed meshcat.Visualizer does not support static_html().")
    path.write_text(visualizer.static_html(), encoding="utf-8")
    return path


def _resolve_record_fps(playback: PlaybackConfig, record_fps: float | None) -> float:
    fps = float(record_fps) if record_fps is not None else (1.0 / max(playback.step_dt, 1e-9))
    if fps <= 0:
        raise ValueError(f"record_fps must be positive. Got {record_fps}.")
    return fps


def _resolve_record_size(record_size: tuple[int, int] | None) -> tuple[int, int]:
    if record_size is None:
        return 1280, 720
    width, height = record_size
    if width <= 0 or height <= 0:
        raise ValueError(f"record_size must contain positive integers. Got {record_size}.")
    return int(width), int(height)


def _resolve_record_targets(record_path: str | pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    path = pathlib.Path(record_path)
    suffix = path.suffix.lower()
    if suffix == "":
        return path.with_suffix(".mp4"), path.with_suffix(".html")
    if suffix == ".html":
        return path.with_suffix(".mp4"), path
    return path, path.with_suffix(".html")


def _find_browser_executable() -> str:
    for env_name in ("EI_VO_BROWSER", "MESHCAT_BROWSER"):
        configured = os.environ.get(env_name)
        if configured:
            return configured

    for candidate in (
        "google-chrome",
        "google-chrome-stable",
        "chromium",
        "chromium-browser",
    ):
        resolved = shutil.which(candidate)
        if resolved is not None:
            return resolved

    raise RuntimeError(
        "MeshCat video recording requires a Chromium-based browser. "
        "Set EI_VO_BROWSER to an executable path or install google-chrome/chromium."
    )


_CAPTURE_QUERY_SCRIPT = """
<script>
(function () {
  let captureAttempts = 0;
  function applyCaptureOptions() {
    if (typeof viewer === "undefined") {
      requestAnimationFrame(applyCaptureOptions);
      return;
    }
    const params = new URLSearchParams(window.location.search);
    if (params.get("meshcat_hide_gui") === "1" && viewer.gui && viewer.gui.domElement) {
      viewer.gui.domElement.style.display = "none";
    }
    const timeValue = params.get("meshcat_time");
    if (timeValue !== null) {
      if (!viewer.animator) {
        if (captureAttempts < 120) {
          captureAttempts += 1;
          requestAnimationFrame(applyCaptureOptions);
        }
        return;
      }
      const actions = viewer.animator.actions || [];
      if (actions.length === 0 && captureAttempts < 120) {
        captureAttempts += 1;
        requestAnimationFrame(applyCaptureOptions);
        return;
      }
      const time = Number(timeValue);
      viewer.animator.reset();
      for (const action of actions) {
        action.play();
        action.paused = true;
        action.enabled = true;
      }
      viewer.animator.pause();
      viewer.animator.seek(Number.isFinite(time) ? time : 0.0);
    }
    if (typeof viewer.set_dirty === "function") {
      viewer.set_dirty();
    }
    if (typeof viewer.render === "function") {
      viewer.render();
    }
  }
  requestAnimationFrame(applyCaptureOptions);
})();
</script>
""".strip()


def _build_capture_html_source(html_source: str) -> str:
    marker = "</body>"
    if marker in html_source:
        return html_source.replace(marker, f"{_CAPTURE_QUERY_SCRIPT}\n{marker}", 1)
    return f"{html_source}\n{_CAPTURE_QUERY_SCRIPT}\n"


def _capture_html_frame(
    browser_path: str,
    html_path: pathlib.Path,
    screenshot_path: pathlib.Path,
    *,
    time_seconds: float,
    size: tuple[int, int],
) -> None:
    width, height = size
    html_url = f"{html_path.resolve().as_uri()}?meshcat_time={time_seconds:.12f}&meshcat_hide_gui=1"
    command = [
        browser_path,
        "--headless",
        "--disable-gpu",
        "--hide-scrollbars",
        "--run-all-compositor-stages-before-draw",
        "--no-first-run",
        "--no-default-browser-check",
        "--allow-file-access-from-files",
        f"--window-size={width},{height}",
        "--virtual-time-budget=1000",
        f"--screenshot={screenshot_path.as_posix()}",
        html_url,
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        details = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"MeshCat video capture failed for frame at t={time_seconds:.6f}s: {details}")
    if not screenshot_path.exists():
        raise RuntimeError(f"MeshCat video capture did not create {screenshot_path}.")


def _coerce_video_frame(frame) -> np.ndarray:
    image = np.asarray(frame)
    if image.ndim == 2:
        image = np.repeat(image[:, :, None], 3, axis=2)
    if image.ndim == 3 and image.shape[2] == 4:
        image = image[:, :, :3]
    if image.dtype != np.uint8:
        scale = 255.0 if np.issubdtype(image.dtype, np.floating) and float(np.max(image)) <= 1.0 else 1.0
        image = np.clip(image * scale, 0.0, 255.0).astype(np.uint8)
    return image


def _render_video_from_html(
    html_path: pathlib.Path,
    video_path: pathlib.Path,
    *,
    fps: float,
    size: tuple[int, int],
    frame_count: int,
) -> pathlib.Path:
    imageio = _import_imageio()
    browser_path = _find_browser_executable()
    video_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="ei_vo_meshcat_") as temp_dir_name:
        temp_dir = pathlib.Path(temp_dir_name)
        capture_html_path = temp_dir / "capture.html"
        capture_html_path.write_text(
            _build_capture_html_source(html_path.read_text(encoding="utf-8")),
            encoding="utf-8",
        )
        try:
            writer = imageio.get_writer(video_path.as_posix(), fps=fps)
        except ValueError as exc:
            raise RuntimeError(
                "MeshCat MP4 recording requires an imageio video backend. "
                "Install 'imageio[ffmpeg]' (recommended) or 'imageio[pyav]'."
            ) from exc
        try:
            for frame_index in range(frame_count):
                screenshot_path = temp_dir / f"{frame_index:07d}.png"
                _capture_html_frame(
                    browser_path,
                    capture_html_path,
                    screenshot_path,
                    time_seconds=frame_index / fps,
                    size=size,
                )
                writer.append_data(_coerce_video_frame(imageio.imread(screenshot_path.as_posix())))
        finally:
            writer.close()

    return video_path


def _export_recording_artifacts(
    visualizer,
    *,
    playback: PlaybackConfig,
    record_path: str | pathlib.Path,
    record_fps: float | None,
    record_size: tuple[int, int] | None,
    frame_count: int,
    recorder: "_AnimationRecorder | None",
) -> tuple[pathlib.Path, pathlib.Path]:
    video_path, html_path = _resolve_record_targets(record_path)
    if recorder is not None:
        recorder.begin_frame(None)
        recorder.apply(visualizer)
    saved_html = _save_html(visualizer, html_path)
    _render_video_from_html(
        saved_html,
        video_path,
        fps=_resolve_record_fps(playback, record_fps),
        size=_resolve_record_size(record_size),
        frame_count=frame_count,
    )
    return video_path, saved_html


class _AnimationRecorder:
    def __init__(self, meshcat_module, *, fps: float) -> None:
        animation_module = getattr(meshcat_module, "animation", None)
        animation_cls = getattr(animation_module, "Animation", None) if animation_module is not None else None
        self.animation = None if animation_cls is None else animation_cls(default_framerate=fps)
        self._frame_index: int | None = None

    @property
    def enabled(self) -> bool:
        return self.animation is not None

    def begin_frame(self, frame_index: int | None) -> None:
        self._frame_index = None if frame_index is None else int(frame_index)

    def capture_transform(self, node, transform: np.ndarray) -> None:
        if self.animation is None or self._frame_index is None:
            return
        with self.animation.at_frame(node, self._frame_index) as frame_visualizer:
            frame_visualizer.set_transform(np.asarray(transform, dtype=float))

    def apply(self, visualizer) -> bool:
        if self.animation is None:
            return False
        visualizer.set_animation(self.animation, play=True, repetitions=1)
        return True


class _RecordingNode:
    def __init__(self, node, recorder: _AnimationRecorder | None) -> None:
        self._node = node
        self._recorder = recorder

    @property
    def path(self):
        return self._node.path

    def __getitem__(self, key):
        return _RecordingNode(self._node[key], self._recorder)

    def set_transform(self, matrix=np.eye(4)):
        self._node.set_transform(matrix)
        if self._recorder is not None:
            self._recorder.capture_transform(self, matrix)

    def __getattr__(self, name):
        return getattr(self._node, name)


def _build_recording_visualizer(meshcat_module, playback: PlaybackConfig, *, record_fps: float | None):
    recorder = _AnimationRecorder(meshcat_module, fps=_resolve_record_fps(playback, record_fps))
    return recorder if recorder.enabled else None


def _play_with_mujoco(
    path: pathlib.Path,
    trajectory: Trajectory,
    playback: PlaybackConfig,
    *,
    camera: CameraSettings | Mapping[str, object] | None,
    open_browser: bool,
    root_path: str,
    record_path: str | pathlib.Path | None,
    record_fps: float | None,
    record_size: tuple[int, int] | None,
) -> None:
    mj = _import_mujoco()
    meshcat = _import_meshcat()

    model = mj.MjModel.from_xml_path(path.as_posix())
    data = mj.MjData(model)
    arm_joints = detect_arm_joints(model, expected_dof=trajectory.dof)
    positions = _clip_positions_to_limits(trajectory.q, arm_joints.limits)

    visualizer = meshcat.Visualizer()
    recorder = _build_recording_visualizer(meshcat, playback, record_fps=record_fps) if record_path else None
    recording_visualizer = _RecordingNode(visualizer, recorder) if recorder is not None else visualizer
    if open_browser and hasattr(recording_visualizer, "open"):
        recording_visualizer.open()
    _apply_camera_settings(recording_visualizer, camera)
    scene_root = recording_visualizer[root_path] if root_path else recording_visualizer
    geom_ids = _create_scene(scene_root, model)

    exported = False
    while True:
        for frame_index, row in enumerate(positions):
            if recorder is not None:
                recorder.begin_frame(frame_index if not exported else None)
            for qpos_address, value in zip(arm_joints.qpos_addresses, row):
                data.qpos[qpos_address] = float(value)
            mj.mj_forward(model, data)
            _update_scene(scene_root, data, geom_ids)
            time.sleep(playback.step_dt)

        if record_path is not None and not exported:
            _export_recording_artifacts(
                recording_visualizer,
                playback=playback,
                record_path=record_path,
                record_fps=record_fps,
                record_size=record_size,
                frame_count=len(positions),
                recorder=recorder,
            )
            exported = True

        if not playback.loop:
            break


def _play_with_pinocchio(
    path: pathlib.Path,
    trajectory: Trajectory,
    playback: PlaybackConfig,
    *,
    camera: CameraSettings | Mapping[str, object] | None,
    open_browser: bool,
    root_path: str,
    record_path: str | pathlib.Path | None,
    record_fps: float | None,
    record_size: tuple[int, int] | None,
) -> None:
    meshcat = _import_meshcat()
    pin, MeshcatVisualizer = _import_pinocchio_visualizer()

    model, collision_model, visual_model = pin.buildModelsFromUrdf(path.as_posix())
    model_dof = int(getattr(model, "nq"))
    if model_dof != trajectory.dof:
        raise ValueError(
            f"Trajectory dof ({trajectory.dof}) does not match Pinocchio model dof ({model_dof})."
        )

    lower_limits = getattr(model, "lowerPositionLimit", np.full(model_dof, -np.inf))
    upper_limits = getattr(model, "upperPositionLimit", np.full(model_dof, np.inf))
    positions = _clip_positions_to_pinocchio_limits(trajectory.q, lower_limits, upper_limits)

    visualizer = meshcat.Visualizer()
    recorder = _build_recording_visualizer(meshcat, playback, record_fps=record_fps) if record_path else None
    recording_visualizer = _RecordingNode(visualizer, recorder) if recorder is not None else visualizer
    visualizer_wrapper = MeshcatVisualizer(model, collision_model, visual_model)
    visualizer_wrapper.initViewer(viewer=recording_visualizer, open=open_browser, loadModel=False)
    _apply_camera_settings(recording_visualizer, camera)
    visualizer_wrapper.loadViewerModel(rootNodeName=root_path or "pinocchio")

    exported = False
    while True:
        for frame_index, row in enumerate(positions):
            if recorder is not None:
                recorder.begin_frame(frame_index if not exported else None)
            visualizer_wrapper.display(np.asarray(row, dtype=float))
            time.sleep(playback.step_dt)

        if record_path is not None and not exported:
            _export_recording_artifacts(
                recording_visualizer,
                playback=playback,
                record_path=record_path,
                record_fps=record_fps,
                record_size=record_size,
                frame_count=len(positions),
                recorder=recorder,
            )
            exported = True

        if not playback.loop:
            break


def play(
    model_path: str | pathlib.Path | None,
    traj,
    slow: float = 1.0,
    hz: float = 240.0,
    camera=None,
    loop: bool = False,
    record_path: str | pathlib.Path | None = None,
    record_fps: float | None = None,
    record_size: tuple[int, int] | None = None,
    *,
    open_browser: bool = True,
    root_path: str = "",
    kinematics_backend: str | None = None,
    kinematics_model_path: str | pathlib.Path | None = None,
    base_link: str | None = None,
    end_link: str | None = None,
) -> None:
    """Render a trajectory in MeshCat.

    URDF inputs are rendered via Pinocchio's visual model so `<visual>` geometry
    shows up in MeshCat. Other inputs fall back to MuJoCo geom playback. When
    ``record_path`` is given, MeshCat writes an MP4 recording plus a standalone
    HTML sidecar after the first playback pass. The exported HTML includes the
    same keyframed playback animation used for the MP4 conversion.
    """

    del kinematics_backend, kinematics_model_path, base_link, end_link

    if model_path is None:
        raise ValueError("--model is required when using the meshcat renderer.")

    playback = PlaybackConfig(hz=hz, slow=slow, loop=loop)
    path = pathlib.Path(model_path)
    trajectory = Trajectory.coerce(traj)
    if path.suffix.lower() == ".urdf":
        _play_with_pinocchio(
            path,
            trajectory,
            playback,
            camera=camera,
            open_browser=open_browser,
            root_path=root_path,
            record_path=record_path,
            record_fps=record_fps,
            record_size=record_size,
        )
        return

    _play_with_mujoco(
        path,
        trajectory,
        playback,
        camera=camera,
        open_browser=open_browser,
        root_path=root_path,
        record_path=record_path,
        record_fps=record_fps,
        record_size=record_size,
    )


__all__ = ["play"]
