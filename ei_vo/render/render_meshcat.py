"""MeshCat-based playback backend for URDF models."""

from __future__ import annotations

import pathlib
import time
import webbrowser
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from ..config import (
    CameraSettings,
    PlaybackConfig,
    RecordingConfig,
    coerce_camera_settings,
    coerce_playback_config,
    normalize_runtime_config,
    resolve_recording_fps,
)
from ..core import Trajectory
from ..modeling import load_robot_model
from ._urdfpy import install_urdfpy_compat_shims


@dataclass(slots=True)
class _PreparedVisual:
    link: object
    geometry: object
    material: object | None
    local_pose: np.ndarray
    path: str


def _import_meshcat():
    try:
        import meshcat
        import meshcat.geometry as meshcat_geometry
    except ImportError as exc:
        raise RuntimeError("The 'meshcat' renderer requires the optional dependency 'meshcat'.") from exc
    return meshcat, meshcat_geometry


def _import_urdfpy():
    install_urdfpy_compat_shims()
    try:
        from urdfpy import URDF
    except ImportError as exc:
        raise RuntimeError(
            "The 'meshcat' renderer requires the optional dependency 'urdfpy'. "
            "Install it with `uv sync --extra meshcat`."
        ) from exc
    return URDF


def _create_visualizer(meshcat_module):
    return meshcat_module.Visualizer()


def _require_urdf_model_path(model_path: str | pathlib.Path | None) -> pathlib.Path:
    if model_path is None:
        raise ValueError("--model is required when using the meshcat renderer.")
    path = pathlib.Path(model_path)
    if path.suffix.lower() != ".urdf":
        raise ValueError(f"The 'meshcat' renderer only supports URDF models. Got {path!s}.")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path.resolve()


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
_URDF_CYLINDER_TO_MESHCAT = _rotation_x(np.pi / 2.0)


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
    settings = coerce_camera_settings(camera)
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


def _save_html(visualizer, record_path: str | pathlib.Path) -> pathlib.Path:
    path = pathlib.Path(record_path)
    if path.suffix == "":
        path = path.with_suffix(".html")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not hasattr(visualizer, "static_html"):
        raise RuntimeError("The installed meshcat.Visualizer does not support static_html().")
    path.write_text(visualizer.static_html(), encoding="utf-8")
    return path


def _resolve_record_html_path(record_path: str | pathlib.Path) -> pathlib.Path:
    path = pathlib.Path(record_path)
    return path if path.suffix.lower() == ".html" else path.with_suffix(".html")


def _export_recording_artifacts(
    visualizer,
    *,
    recording: RecordingConfig,
    recorder: "_AnimationRecorder | None",
) -> pathlib.Path:
    if recorder is not None:
        recorder.begin_frame(None)
        recorder.apply(visualizer)
    return _save_html(visualizer, _resolve_record_html_path(recording.path))


def _open_standalone_recording_html(html_path: pathlib.Path) -> None:
    try:
        webbrowser.open(html_path.resolve(strict=False).as_uri(), new=2)
    except Exception:
        return


def _viewer_url(visualizer) -> str | None:
    url_method = getattr(visualizer, "url", None)
    if callable(url_method):
        try:
            url = url_method()
        except Exception:
            url = None
        if url:
            return str(url)

    window = getattr(visualizer, "window", None)
    web_url = getattr(window, "web_url", None)
    return None if web_url is None else str(web_url)


def _report_live_viewer_ready(visualizer, *, visual_count: int) -> None:
    viewer_url = _viewer_url(visualizer)
    if viewer_url is not None:
        print(f"[ei-vo] meshcat viewer ready: {viewer_url}")
    print(f"[ei-vo] loaded {visual_count} visual object(s) into the scene")
    print(
        "[ei-vo] if the page is blank, open the URL on the same machine. "
        "For SSH / remote IDE sessions, forward that port first."
    )


def _wait_until_interrupted(interval_s: float = 0.5) -> None:
    try:
        while True:
            time.sleep(interval_s)
    except KeyboardInterrupt:
        return


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


def _build_recording_visualizer(
    meshcat_module,
    playback: PlaybackConfig,
    *,
    recording: RecordingConfig,
):
    recorder = _AnimationRecorder(meshcat_module, fps=resolve_recording_fps(playback, recording))
    return recorder if recorder.enabled else None


def _material_rgba(material) -> np.ndarray | None:
    if material is None:
        return None
    color = getattr(material, "color", None)
    if color is None:
        return None
    rgba = np.asarray(color, dtype=float)
    if rgba.shape != (4,):
        return None
    return np.clip(rgba, 0.0, 1.0)


def _rgba_to_hex(rgba: np.ndarray) -> int:
    rgb = np.clip(np.round(np.asarray(rgba[:3], dtype=float) * 255.0), 0.0, 255.0).astype(np.uint8)
    return (int(rgb[0]) << 16) | (int(rgb[1]) << 8) | int(rgb[2])


def _make_material(meshcat_geometry, rgba: np.ndarray | None):
    if rgba is None:
        return None
    opacity = float(rgba[3])
    return meshcat_geometry.MeshLambertMaterial(
        color=_rgba_to_hex(rgba),
        opacity=opacity,
        transparent=opacity < 0.999,
    )


def _geometry_scale_pose(geometry) -> np.ndarray:
    mesh = getattr(geometry, "mesh", None)
    scale = getattr(mesh, "scale", None)
    pose = np.eye(4, dtype=float)
    if scale is not None:
        pose[:3, :3] = np.diag(np.asarray(scale, dtype=float))
    return pose


def _mesh_to_triangular_geometry(meshcat_geometry, mesh):
    vertices = np.asarray(getattr(mesh, "vertices", ()), dtype=float)
    faces = np.asarray(getattr(mesh, "faces", ()), dtype=np.uint32)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or vertices.shape[0] == 0:
        raise RuntimeError("URDF mesh geometry does not provide valid vertices for MeshCat rendering.")
    if faces.ndim != 2 or faces.shape[1] != 3 or faces.shape[0] == 0:
        raise RuntimeError("URDF mesh geometry does not provide triangular faces for MeshCat rendering.")
    return meshcat_geometry.TriangularMeshGeometry(vertices, faces)


def _geometry_objects(meshcat_geometry, geometry) -> tuple[list[object], np.ndarray]:
    if getattr(geometry, "box", None) is not None:
        return [meshcat_geometry.Box(np.asarray(geometry.box.size, dtype=float))], np.eye(4, dtype=float)
    if getattr(geometry, "cylinder", None) is not None:
        return [
            meshcat_geometry.Cylinder(
                float(geometry.cylinder.length),
                radius=float(geometry.cylinder.radius),
            )
        ], _make_transform(np.zeros(3, dtype=float), rotation=_URDF_CYLINDER_TO_MESHCAT)
    if getattr(geometry, "sphere", None) is not None:
        return [meshcat_geometry.Sphere(float(geometry.sphere.radius))], np.eye(4, dtype=float)

    mesh_geometry = getattr(geometry, "mesh", None)
    if mesh_geometry is not None:
        meshes = getattr(mesh_geometry, "meshes", ())
        if not meshes:
            raise RuntimeError("URDF mesh geometry did not provide any meshes for MeshCat rendering.")
        return [
            _mesh_to_triangular_geometry(meshcat_geometry, mesh)
            for mesh in meshes
        ], _geometry_scale_pose(geometry)

    raise RuntimeError("Unsupported URDF visual geometry for the meshcat renderer.")


def _scene_root_name(root_path: str) -> str:
    normalized = root_path.strip("/")
    return normalized or "robot"


def _visual_path(root_name: str, link_name: str, counts: dict[str, int]) -> str:
    index = counts.get(link_name, 0)
    counts[link_name] = index + 1
    suffix = "" if index == 0 else f"_{index}"
    return f"{root_name}/visuals/{link_name}{suffix}"


def _prepare_visuals(robot, meshcat_geometry, *, root_name: str) -> tuple[_PreparedVisual, ...]:
    visuals: list[_PreparedVisual] = []
    counts: dict[str, int] = {}
    for link in getattr(robot, "links", ()):
        link_name = getattr(link, "name", None) or "link"
        for visual in getattr(link, "visuals", ()):
            geometry_objects, geometry_pose = _geometry_objects(meshcat_geometry, visual.geometry)
            local_pose = np.asarray(getattr(visual, "origin", np.eye(4, dtype=float)), dtype=float).dot(geometry_pose)
            material = _make_material(meshcat_geometry, _material_rgba(getattr(visual, "material", None)))
            for geometry_object in geometry_objects:
                visuals.append(
                    _PreparedVisual(
                        link=link,
                        geometry=geometry_object,
                        material=material,
                        local_pose=local_pose.copy(),
                        path=_visual_path(root_name, link_name, counts),
                    )
                )

    if not visuals:
        raise RuntimeError("The URDF model does not contain visual geometry.")
    return tuple(visuals)


def _config_from_row(joint_names: tuple[str, ...], row: np.ndarray) -> dict[str, float]:
    return {joint_name: float(value) for joint_name, value in zip(joint_names, row)}


def _load_visual_objects(visualizer, visuals: tuple[_PreparedVisual, ...], *, root_name: str) -> None:
    visualizer[f"{root_name}/collisions"].set_property("visible", False)
    visualizer[f"{root_name}/visuals"].set_property("visible", True)
    for visual in visuals:
        visualizer[visual.path].set_object(visual.geometry, visual.material)


def _display_row(
    visualizer,
    robot,
    visuals: tuple[_PreparedVisual, ...],
    joint_names: tuple[str, ...],
    row: np.ndarray,
) -> None:
    fk = robot.link_fk(cfg=_config_from_row(joint_names, row))
    for visual in visuals:
        visualizer[visual.path].set_transform(np.asarray(fk[visual.link], dtype=float).dot(visual.local_pose))


def _play_with_urdfpy(
    path: pathlib.Path,
    trajectory: Trajectory,
    playback: PlaybackConfig,
    *,
    camera: CameraSettings | Mapping[str, object] | None,
    open_browser: bool,
    hold_open: bool,
    root_path: str,
    recording: RecordingConfig | None,
) -> None:
    meshcat, meshcat_geometry = _import_meshcat()
    URDF = _import_urdfpy()

    robot_model = load_robot_model(path, expected_dof=trajectory.dof)
    positions = robot_model.clamp(trajectory.q)
    urdf_robot = URDF.load(path.as_posix())
    root_name = _scene_root_name(root_path)
    visuals = _prepare_visuals(urdf_robot, meshcat_geometry, root_name=root_name)

    visualizer = _create_visualizer(meshcat)
    recorder = _build_recording_visualizer(meshcat, playback, recording=recording) if recording else None
    recording_visualizer = _RecordingNode(visualizer, recorder) if recorder is not None else visualizer
    _load_visual_objects(recording_visualizer, visuals, root_name=root_name)
    _apply_camera_settings(recording_visualizer, camera)
    if recording is None:
        _report_live_viewer_ready(visualizer, visual_count=len(visuals))
    if open_browser and recording is None:
        visualizer.open()

    exported = False
    while True:
        for frame_index, row in enumerate(positions):
            if recorder is not None:
                recorder.begin_frame(frame_index if not exported else None)
            _display_row(recording_visualizer, urdf_robot, visuals, robot_model.joint_names, row)
            time.sleep(playback.step_dt)

        if recording is not None and not exported:
            saved_html = _export_recording_artifacts(
                recording_visualizer,
                recording=recording,
                recorder=recorder,
            )
            if open_browser:
                _open_standalone_recording_html(saved_html)
            exported = True

        if not playback.loop:
            if hold_open and recording is None:
                _wait_until_interrupted()
            break


def play_trajectory(
    model_path: str | pathlib.Path | None,
    trajectory: Trajectory | np.ndarray | list[list[float]] | list[float],
    *,
    playback: PlaybackConfig | None = None,
    camera: CameraSettings | Mapping[str, object] | None = None,
    recording: RecordingConfig | None = None,
    open_browser: bool = True,
    hold_open: bool = False,
    root_path: str = "",
) -> None:
    """Render a URDF trajectory in MeshCat using normalized runtime config objects."""

    path = _require_urdf_model_path(model_path)
    playback = coerce_playback_config(playback)
    if recording is not None:
        if recording.size is not None:
            raise ValueError("record_size is not supported for MeshCat HTML export.")
        if recording.frames_dir is not None:
            raise ValueError("record_frames_dir is not supported for MeshCat HTML export.")

    _play_with_urdfpy(
        path,
        Trajectory.coerce(trajectory),
        playback,
        camera=camera,
        open_browser=open_browser,
        hold_open=hold_open,
        root_path=root_path,
        recording=recording,
    )


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
    record_frames_dir: str | pathlib.Path | None = None,
    *,
    open_browser: bool = True,
    hold_open: bool = False,
    root_path: str = "",
    kinematics_backend: str | None = None,
    kinematics_model_path: str | pathlib.Path | None = None,
    base_link: str | None = None,
    end_link: str | None = None,
) -> None:
    """Render a URDF trajectory in MeshCat."""

    del kinematics_backend, kinematics_model_path, base_link, end_link

    playback_config, camera_settings, recording = normalize_runtime_config(
        hz=hz,
        slow=slow,
        loop=loop,
        camera=camera,
        record_path=record_path,
        record_fps=record_fps,
        record_size=record_size,
        record_frames_dir=record_frames_dir,
    )
    play_trajectory(
        model_path,
        traj,
        playback=playback_config,
        camera=camera_settings,
        recording=recording,
        open_browser=open_browser,
        hold_open=hold_open,
        root_path=root_path,
    )


__all__ = ["play", "play_trajectory"]
