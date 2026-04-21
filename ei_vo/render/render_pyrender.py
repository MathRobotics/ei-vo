"""Offscreen URDF rendering with pyrender and urdfpy."""

from __future__ import annotations

import collections
import collections.abc
import contextlib
import fractions
import math
import os
import pathlib
import re
import sys
import types
from dataclasses import dataclass

import numpy as np

from ..config import (
    CameraSettings,
    normalize_runtime_config,
    resolve_recording_fps,
)
from ..core import FrameSequenceWriter, Trajectory
from ..modeling import load_robot_model

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".ppm"}
_VIDEO_SUFFIXES = {".gif", ".mov", ".mp4", ".webm"}
_DEFAULT_RECORD_SIZE = (1280, 720)
_DEFAULT_CAMERA_AZIMUTH = 45.0
_DEFAULT_CAMERA_ELEVATION = 25.0
_DEFAULT_CAMERA_YFOV = math.radians(45.0)
_FORWARDED_DISPLAY_PREFIXES = ("localhost:", "127.0.0.1:", "[::1]:")


@dataclass(slots=True)
class _PreparedVisual:
    link: object
    mesh: object
    local_pose: np.ndarray


def _require_urdf_model_path(model_path: str | pathlib.Path | None) -> pathlib.Path:
    if model_path is None:
        raise ValueError("--model is required when using the pyrender renderer.")
    path = pathlib.Path(model_path)
    if path.suffix.lower() != ".urdf":
        raise ValueError(f"The 'pyrender' renderer only supports URDF models. Got {path!s}.")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path.resolve()


def _parse_version_fallback(value: object):
    text = str(value)
    parts: list[tuple[int, object]] = []
    for part in re.split(r"[^0-9A-Za-z]+", text):
        if not part:
            continue
        if part.isdigit():
            parts.append((0, int(part)))
        else:
            parts.append((1, part.lower()))
    return tuple(parts)


def _install_urdfpy_compat_shims() -> None:
    """Patch stdlib and NumPy symbols needed by urdfpy's pinned networkx 2.2."""

    for name in ("Iterable", "Mapping", "MutableMapping", "Sequence", "Set"):
        if not hasattr(collections, name):
            setattr(collections, name, getattr(collections.abc, name))

    if not hasattr(fractions, "gcd"):
        fractions.gcd = math.gcd  # type: ignore[attr-defined]

    numpy_aliases = {
        "bool": bool,
        "complex_": np.complex128,
        "float": float,
        "float_": np.float64,
        "int": int,
        "int_": np.int64,
        "object": object,
        "unicode_": np.str_,
    }
    for name, value in numpy_aliases.items():
        if name not in np.__dict__:
            setattr(np, name, value)

    if "pkg_resources" not in sys.modules:
        pkg_resources = types.ModuleType("pkg_resources")

        def parse_version(value: object):
            try:
                from packaging.version import parse as packaging_parse_version
            except Exception:
                return _parse_version_fallback(value)
            return packaging_parse_version(str(value))

        pkg_resources.parse_version = parse_version  # type: ignore[attr-defined]
        sys.modules["pkg_resources"] = pkg_resources


def _is_forwarded_x11_display(display: str | None) -> bool:
    if display is None:
        return False
    normalized = display.strip().lower()
    return any(normalized.startswith(prefix) for prefix in _FORWARDED_DISPLAY_PREFIXES)


def _should_auto_use_egl() -> bool:
    if sys.platform != "linux":
        return False
    if os.environ.get("PYOPENGL_PLATFORM"):
        return False

    display = os.environ.get("DISPLAY")
    wayland_display = os.environ.get("WAYLAND_DISPLAY")
    ssh_active = bool(os.environ.get("SSH_CONNECTION") or os.environ.get("SSH_CLIENT"))
    if not display and not wayland_display:
        return True
    if _is_forwarded_x11_display(display):
        return True
    if ssh_active and os.environ.get("XDG_SESSION_TYPE") == "tty":
        return True
    return False


def _prepare_pyopengl_platform() -> dict[str, object]:
    platform = os.environ.get("PYOPENGL_PLATFORM")
    auto_selected = False
    if platform is None and _should_auto_use_egl():
        os.environ["PYOPENGL_PLATFORM"] = "egl"
        platform = "egl"
        auto_selected = True
    return {
        "platform": None if platform is None else platform.strip().lower(),
        "auto_selected": auto_selected,
    }


def _import_runtime_dependencies():
    runtime = _prepare_pyopengl_platform()
    _install_urdfpy_compat_shims()
    try:
        import pyrender
        import trimesh
        from urdfpy import URDF
    except ImportError as exc:
        raise RuntimeError(
            "The 'pyrender' renderer requires the optional dependencies 'pyrender' "
            "and 'urdfpy'. Install them with `uv sync --extra pyrender`."
        ) from exc
    return pyrender, URDF, trimesh, runtime


@contextlib.contextmanager
def _prefer_default_egl_display(pyrender, runtime: dict[str, object]):
    if runtime.get("platform") != "egl":
        yield
        return
    if os.environ.get("EGL_DEVICE_ID"):
        yield
        return

    egl_module = getattr(getattr(pyrender, "platforms", None), "egl", None)
    if egl_module is None:
        try:
            from pyrender.platforms import egl as egl_module
        except Exception:
            yield
            return

    get_device_by_index = getattr(egl_module, "get_device_by_index", None)
    egl_device = getattr(egl_module, "EGLDevice", None)
    if not callable(get_device_by_index) or egl_device is None:
        yield
        return

    original = get_device_by_index
    egl_module.get_device_by_index = lambda device_id: egl_device(None)
    try:
        yield
    finally:
        egl_module.get_device_by_index = original


def _raise_offscreen_context_error(exc: Exception, runtime: dict[str, object]) -> None:
    platform = runtime.get("platform")
    if platform == "egl":
        raise RuntimeError(
            "The 'pyrender' renderer could not create an EGL offscreen OpenGL context. "
            "Install Mesa EGL or OSMesa, and set `PYOPENGL_PLATFORM=osmesa` if your host "
            "does not expose a usable EGL default display."
        ) from exc

    raise RuntimeError(
        "The 'pyrender' renderer could not create an offscreen OpenGL context. "
        "Install EGL or OSMesa and, if needed, set PYOPENGL_PLATFORM=egl or osmesa."
    ) from exc


def _resolve_record_output_path(record_path: str | pathlib.Path) -> tuple[pathlib.Path, str]:
    output_path = pathlib.Path(record_path)
    suffix = output_path.suffix.lower()
    if suffix in _VIDEO_SUFFIXES:
        return output_path, "video"
    if suffix in _IMAGE_SUFFIXES:
        return output_path, "image"
    if suffix == "":
        return output_path.with_suffix(".mp4"), "video"
    raise ValueError(
        "The 'pyrender' renderer supports image outputs "
        f"{sorted(_IMAGE_SUFFIXES)} and video outputs {sorted(_VIDEO_SUFFIXES)}. "
        f"Got {output_path.suffix or '<none>'!r}."
    )


def _coerce_record_size(size: tuple[int, int] | None) -> tuple[int, int]:
    if size is None:
        return _DEFAULT_RECORD_SIZE
    width, height = size
    if width <= 0 or height <= 0:
        raise ValueError(f"record_size must contain positive integers. Got {size}.")
    return int(width), int(height)


def _config_from_row(joint_names: tuple[str, ...], row: np.ndarray) -> dict[str, float]:
    return {joint_name: float(value) for joint_name, value in zip(joint_names, row)}


def _bounds_corners(bounds: np.ndarray) -> np.ndarray:
    mins = bounds[0]
    maxs = bounds[1]
    return np.array(
        [
            [mins[0], mins[1], mins[2]],
            [mins[0], mins[1], maxs[2]],
            [mins[0], maxs[1], mins[2]],
            [mins[0], maxs[1], maxs[2]],
            [maxs[0], mins[1], mins[2]],
            [maxs[0], mins[1], maxs[2]],
            [maxs[0], maxs[1], mins[2]],
            [maxs[0], maxs[1], maxs[2]],
        ],
        dtype=float,
    )


def _mesh_world_bounds(mesh, pose: np.ndarray) -> np.ndarray:
    bounds = np.asarray(getattr(mesh, "bounds", np.zeros((2, 3), dtype=float)), dtype=float)
    if bounds.shape != (2, 3) or not np.all(np.isfinite(bounds)):
        center = np.asarray(pose[:3, 3], dtype=float)
        return np.vstack((center, center))

    corners = _bounds_corners(bounds)
    rotation = np.asarray(pose[:3, :3], dtype=float)
    translation = np.asarray(pose[:3, 3], dtype=float)
    world_corners = corners @ rotation.T + translation
    return np.vstack((np.min(world_corners, axis=0), np.max(world_corners, axis=0)))


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


def _apply_material_color(mesh, rgba: np.ndarray | None):
    if rgba is None:
        return mesh.copy()

    colored = mesh.copy()
    color = np.clip(np.round(rgba * 255.0), 0.0, 255.0).astype(np.uint8)
    visual = getattr(colored, "visual", None)
    face_count = len(getattr(colored, "faces", ()))
    vertex_count = len(getattr(colored, "vertices", ()))
    if visual is not None and face_count > 0 and hasattr(visual, "face_colors"):
        visual.face_colors = np.tile(color, (face_count, 1))
    elif visual is not None and vertex_count > 0 and hasattr(visual, "vertex_colors"):
        visual.vertex_colors = np.tile(color, (vertex_count, 1))
    return colored


def _geometry_scale_pose(geometry) -> np.ndarray:
    mesh = getattr(geometry, "mesh", None)
    scale = getattr(mesh, "scale", None)
    pose = np.eye(4, dtype=float)
    if scale is not None:
        pose[:3, :3] = np.diag(np.asarray(scale, dtype=float))
    return pose


def _geometry_meshes(trimesh, geometry, material) -> tuple[list[object], np.ndarray]:
    if getattr(geometry, "box", None) is not None:
        meshes = [trimesh.creation.box(extents=np.asarray(geometry.box.size, dtype=float))]
        return [_apply_material_color(mesh, _material_rgba(material)) for mesh in meshes], np.eye(4, dtype=float)
    if getattr(geometry, "cylinder", None) is not None:
        meshes = [
            trimesh.creation.cylinder(
                radius=float(geometry.cylinder.radius),
                height=float(geometry.cylinder.length),
            )
        ]
        return [_apply_material_color(mesh, _material_rgba(material)) for mesh in meshes], np.eye(4, dtype=float)
    if getattr(geometry, "sphere", None) is not None:
        meshes = [trimesh.creation.icosphere(radius=float(geometry.sphere.radius))]
        return [_apply_material_color(mesh, _material_rgba(material)) for mesh in meshes], np.eye(4, dtype=float)

    mesh_geometry = getattr(geometry, "mesh", None)
    if mesh_geometry is not None:
        meshes = [_apply_material_color(mesh, _material_rgba(material)) for mesh in mesh_geometry.meshes]
        return meshes, _geometry_scale_pose(geometry)

    raise RuntimeError("Unsupported URDF visual geometry for the pyrender renderer.")


def _prepare_visuals(robot, trimesh) -> tuple[_PreparedVisual, ...]:
    visuals: list[_PreparedVisual] = []
    for link in getattr(robot, "links", ()):
        for visual in getattr(link, "visuals", ()):
            meshes, geometry_pose = _geometry_meshes(trimesh, visual.geometry, visual.material)
            local_pose = np.asarray(visual.origin, dtype=float).dot(geometry_pose)
            for mesh in meshes:
                visuals.append(_PreparedVisual(link=link, mesh=mesh, local_pose=local_pose.copy()))

    if not visuals:
        raise RuntimeError("The URDF model does not contain visual geometry.")
    return tuple(visuals)


def _build_pose_cache(robot, visuals: tuple[_PreparedVisual, ...], joint_names: tuple[str, ...], positions: np.ndarray):
    frame_poses: list[tuple[np.ndarray, ...]] = []
    min_bounds: list[np.ndarray] = []
    max_bounds: list[np.ndarray] = []

    for row in positions:
        fk = robot.link_fk(cfg=_config_from_row(joint_names, row))
        poses = tuple(np.asarray(fk[visual.link], dtype=float).dot(visual.local_pose) for visual in visuals)
        frame_poses.append(poses)

        for visual, pose in zip(visuals, poses):
            bounds = _mesh_world_bounds(visual.mesh, pose)
            min_bounds.append(bounds[0])
            max_bounds.append(bounds[1])

    if not frame_poses:
        raise ValueError("Trajectory must contain at least one frame.")

    scene_bounds = np.vstack((np.min(min_bounds, axis=0), np.max(max_bounds, axis=0)))
    return tuple(frame_poses), scene_bounds


def _look_at_pose(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    forward = np.asarray(target - eye, dtype=float)
    forward_norm = np.linalg.norm(forward)
    if forward_norm <= 1e-9:
        raise ValueError("camera eye and lookat must not coincide.")
    forward /= forward_norm

    world_up = np.array([0.0, 0.0, 1.0], dtype=float)
    right = np.cross(forward, world_up)
    right_norm = np.linalg.norm(right)
    if right_norm <= 1e-9:
        world_up = np.array([0.0, 1.0, 0.0], dtype=float)
        right = np.cross(forward, world_up)
        right_norm = np.linalg.norm(right)
    right /= right_norm
    up = np.cross(right, forward)
    up /= max(np.linalg.norm(up), 1e-9)

    pose = np.eye(4, dtype=float)
    pose[:3, 0] = right
    pose[:3, 1] = up
    pose[:3, 2] = -forward
    pose[:3, 3] = eye
    return pose


def _resolve_camera_pose(
    bounds: np.ndarray,
    camera: CameraSettings | None,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    center = 0.5 * (bounds[0] + bounds[1])
    extents = bounds[1] - bounds[0]
    scene_scale = max(float(np.max(extents)), 0.25)

    lookat = center if camera is None or camera.lookat is None else np.asarray(camera.lookat, dtype=float)
    distance = scene_scale * 2.75 if camera is None or camera.distance is None else float(camera.distance)
    if distance <= 0.0:
        raise ValueError(f"camera distance must be positive. Got {distance}.")
    azimuth = _DEFAULT_CAMERA_AZIMUTH if camera is None or camera.azimuth is None else float(camera.azimuth)
    elevation = _DEFAULT_CAMERA_ELEVATION if camera is None or camera.elevation is None else float(camera.elevation)

    azimuth_rad = math.radians(azimuth)
    elevation_rad = math.radians(elevation)
    eye = lookat + distance * np.array(
        [
            math.cos(elevation_rad) * math.cos(azimuth_rad),
            math.cos(elevation_rad) * math.sin(azimuth_rad),
            math.sin(elevation_rad),
        ],
        dtype=float,
    )
    return _look_at_pose(eye, lookat), lookat, distance, scene_scale


def _make_light_pose(lookat: np.ndarray, distance: float, *, azimuth: float, elevation: float) -> np.ndarray:
    azimuth_rad = math.radians(azimuth)
    elevation_rad = math.radians(elevation)
    eye = lookat + distance * np.array(
        [
            math.cos(elevation_rad) * math.cos(azimuth_rad),
            math.cos(elevation_rad) * math.sin(azimuth_rad),
            math.sin(elevation_rad),
        ],
        dtype=float,
    )
    return _look_at_pose(eye, lookat)


def _build_scene(
    pyrender,
    meshes: tuple[object, ...],
    poses: tuple[np.ndarray, ...],
    *,
    camera_pose: np.ndarray,
    lookat: np.ndarray,
    distance: float,
    scene_scale: float,
    viewport_size: tuple[int, int],
):
    scene = pyrender.Scene(
        bg_color=np.array([1.0, 1.0, 1.0, 1.0], dtype=float),
        ambient_light=np.array([0.08, 0.08, 0.08], dtype=float),
    )
    mesh_nodes = [
        scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=False), pose=pose)
        for mesh, pose in zip(meshes, poses)
    ]

    width, height = viewport_size
    near = max(scene_scale * 0.02, 0.01)
    far = max(distance + scene_scale * 6.0, 10.0)
    camera = pyrender.PerspectiveCamera(
        yfov=_DEFAULT_CAMERA_YFOV,
        aspectRatio=float(width) / float(height),
        znear=near,
        zfar=far,
    )
    camera_node = scene.add(camera, pose=camera_pose)
    scene.main_camera_node = camera_node

    key_light = pyrender.DirectionalLight(color=np.ones(3, dtype=float), intensity=3.0)
    fill_light = pyrender.DirectionalLight(color=np.ones(3, dtype=float), intensity=1.5)
    rim_light = pyrender.DirectionalLight(color=np.ones(3, dtype=float), intensity=1.0)
    scene.add(key_light, pose=camera_pose)
    scene.add(fill_light, pose=_make_light_pose(lookat, distance * 1.25, azimuth=-35.0, elevation=55.0))
    scene.add(rim_light, pose=_make_light_pose(lookat, distance * 1.5, azimuth=160.0, elevation=35.0))
    return scene, mesh_nodes


def _apply_frame_poses(scene, mesh_nodes: list[object], poses: tuple[np.ndarray, ...]) -> None:
    for node, pose in zip(mesh_nodes, poses):
        scene.set_pose(node, pose=pose)


def _coerce_rgb_uint8(frame) -> np.ndarray:
    image = np.asarray(frame)
    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError(f"Expected an image shaped like (H, W, 3/4). Got {image.shape}.")
    if image.shape[2] == 4:
        image = image[:, :, :3]
    if image.dtype != np.uint8:
        scale = 255.0 if np.issubdtype(image.dtype, np.floating) and float(np.max(image)) <= 1.0 else 1.0
        image = np.clip(image * scale, 0.0, 255.0).astype(np.uint8)
    return np.ascontiguousarray(image)


def _write_image(path: pathlib.Path, frame) -> pathlib.Path:
    try:
        from matplotlib import image as matplotlib_image
    except ImportError as exc:
        raise RuntimeError("Saving pyrender images requires the base dependency 'matplotlib'.") from exc

    output_path = pathlib.Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    matplotlib_image.imsave(output_path.as_posix(), _coerce_rgb_uint8(frame))
    return output_path


def _delete_renderer(renderer) -> None:
    delete = getattr(renderer, "delete", None)
    if callable(delete):
        delete()
        return
    close = getattr(renderer, "close", None)
    if callable(close):
        close()


def play(
    model_path: str | pathlib.Path | None,
    traj: Trajectory | np.ndarray | list[list[float]] | list[float],
    slow: float = 1.0,
    hz: float = 240.0,
    camera: CameraSettings | dict[str, object] | None = None,
    loop: bool = False,
    record_path: str | pathlib.Path | None = None,
    record_fps: float | None = None,
    record_size: tuple[int, int] | None = None,
    record_frames_dir: str | pathlib.Path | None = None,
    *,
    kinematics_backend: str | None = None,
    kinematics_model_path: str | pathlib.Path | None = None,
    base_link: str | None = None,
    end_link: str | None = None,
) -> None:
    """Render a URDF trajectory offscreen via pyrender."""

    del kinematics_backend, kinematics_model_path, base_link, end_link

    model_path = _require_urdf_model_path(model_path)
    playback, camera_settings, recording = normalize_runtime_config(
        hz=hz,
        slow=slow,
        loop=loop,
        camera=camera,
        record_path=record_path,
        record_fps=record_fps,
        record_size=record_size,
        record_frames_dir=record_frames_dir,
    )
    if recording is None:
        raise ValueError("The 'pyrender' renderer requires record_path because it only supports offscreen export.")
    if playback.loop:
        raise ValueError("loop is not supported for the pyrender renderer.")

    output_path, output_mode = _resolve_record_output_path(recording.path)
    if output_mode == "image" and recording.frames_dir is not None:
        raise ValueError("record_frames_dir is only supported for pyrender video output.")

    pyrender, URDF, trimesh, runtime = _import_runtime_dependencies()
    traj = Trajectory.coerce(traj)
    robot_model = load_robot_model(model_path, expected_dof=traj.dof)
    positions = robot_model.clamp(traj.q)
    urdf_robot = URDF.load(model_path.as_posix())
    visuals = _prepare_visuals(urdf_robot, trimesh)
    frame_poses, scene_bounds = _build_pose_cache(urdf_robot, visuals, robot_model.joint_names, positions)

    viewport_size = _coerce_record_size(recording.size)
    camera_pose, lookat, distance, scene_scale = _resolve_camera_pose(scene_bounds, camera_settings)
    scene, mesh_nodes = _build_scene(
        pyrender,
        tuple(visual.mesh for visual in visuals),
        frame_poses[0],
        camera_pose=camera_pose,
        lookat=lookat,
        distance=distance,
        scene_scale=scene_scale,
        viewport_size=viewport_size,
    )

    width, height = viewport_size
    try:
        with _prefer_default_egl_display(pyrender, runtime):
            renderer = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height)
    except Exception as exc:
        _raise_offscreen_context_error(exc, runtime)

    try:
        if output_mode == "video":
            writer = FrameSequenceWriter(
                output_path,
                fps=resolve_recording_fps(playback, recording),
                frames_dir=recording.frames_dir,
                temp_prefix="ei_vo_pyrender_",
            )
            try:
                for poses in frame_poses:
                    _apply_frame_poses(scene, mesh_nodes, poses)
                    color, _depth = renderer.render(scene)
                    writer.append_data(color)
            finally:
                writer.close()
            return

        _apply_frame_poses(scene, mesh_nodes, frame_poses[-1])
        color, _depth = renderer.render(scene)
        _write_image(output_path, color)
    finally:
        _delete_renderer(renderer)


__all__ = ["play"]
