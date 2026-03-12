"""MeshCat-based playback backend."""

from __future__ import annotations

import pathlib
import time

import numpy as np

from ..core import Trajectory
from .render_mj import PlaybackConfig, detect_arm_joints


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


def _play_with_mujoco(
    path: pathlib.Path,
    trajectory: Trajectory,
    playback: PlaybackConfig,
    *,
    open_browser: bool,
    root_path: str,
    record_path: str | pathlib.Path | None,
) -> None:
    mj = _import_mujoco()
    meshcat = _import_meshcat()

    model = mj.MjModel.from_xml_path(path.as_posix())
    data = mj.MjData(model)
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


def _play_with_pinocchio(
    path: pathlib.Path,
    trajectory: Trajectory,
    playback: PlaybackConfig,
    *,
    open_browser: bool,
    root_path: str,
    record_path: str | pathlib.Path | None,
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
    visualizer_wrapper = MeshcatVisualizer(model, collision_model, visual_model)
    visualizer_wrapper.initViewer(viewer=visualizer, open=open_browser, loadModel=False)
    visualizer_wrapper.loadViewerModel(rootNodeName=root_path or "pinocchio")

    exported = False
    while True:
        for row in positions:
            visualizer_wrapper.display(np.asarray(row, dtype=float))
            time.sleep(playback.step_dt)

        if record_path is not None and not exported:
            _save_html(visualizer, record_path)
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
    ``record_path`` is given, the current scene is exported as HTML after the
    first playback pass.
    """

    del camera, record_fps, record_size, kinematics_backend, kinematics_model_path, base_link, end_link

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
            open_browser=open_browser,
            root_path=root_path,
            record_path=record_path,
        )
        return

    _play_with_mujoco(
        path,
        trajectory,
        playback,
        open_browser=open_browser,
        root_path=root_path,
        record_path=record_path,
    )


__all__ = ["play"]
