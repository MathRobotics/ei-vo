"""MeshCat-based playback backend for MuJoCo models."""

from __future__ import annotations

import pathlib
import time

import mujoco as mj
import numpy as np

from ..core import Trajectory
from .render_mj import PlaybackConfig, detect_arm_joints


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


def _set_geom_object(visualizer, meshcat_geometry, geom_type: int, geom_id: int, geom_size: np.ndarray, rgba: np.ndarray) -> bool:
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
        if _set_geom_object(visualizer, meshcat_geometry, geom_type, geom_id, geom_size, rgba):
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
    """Render a MuJoCo trajectory in MeshCat.

    This backend uses MuJoCo for kinematics and MeshCat for browser-based
    visualization. When ``record_path`` is given, the current scene is exported
    as HTML after the first playback pass.
    """

    del camera, record_fps, record_size, kinematics_backend, kinematics_model_path, base_link, end_link

    if model_path is None:
        raise ValueError("--model is required when using the meshcat renderer.")

    try:
        import meshcat
    except ImportError as exc:
        raise RuntimeError(
            "The 'meshcat' renderer requires the optional dependency 'meshcat'."
        ) from exc

    playback = PlaybackConfig(hz=hz, slow=slow, loop=loop)
    path = pathlib.Path(model_path)
    model = mj.MjModel.from_xml_path(path.as_posix())
    data = mj.MjData(model)
    trajectory = Trajectory.coerce(traj)
    arm_joints = detect_arm_joints(model, expected_dof=trajectory.dof)
    positions = _clip_positions_to_limits(trajectory.q, arm_joints.limits)

    visualizer = meshcat.Visualizer()
    if open_browser and hasattr(visualizer, "open"):
        visualizer.open()
    scene_root = visualizer[root_path] if root_path else visualizer
    geom_ids = _create_scene(scene_root, model)

    exported = False
    while True:
        for row in positions:
            for qpos_address, value in zip(arm_joints.qpos_addresses, row):
                data.qpos[qpos_address] = float(value)
            mj.mj_forward(model, data)
            _update_scene(scene_root, data, geom_ids)
            time.sleep(playback.step_dt)

        if record_path is not None and not exported:
            _save_html(visualizer, record_path)
            exported = True

        if not playback.loop:
            break


__all__ = ["play"]
