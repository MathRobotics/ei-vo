"""Standalone Blender scene builder for ``ei-vo`` offline rendering."""

from __future__ import annotations

import json
import math
import pathlib
import sys
import urllib.parse
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Any

import bpy
from mathutils import Matrix, Vector

_PACKAGE_ROOT = pathlib.Path(__file__).resolve(strict=False).parents[2]
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))

from ei_vo.render.blender_camera import default_blender_camera_distance

_DEFAULT_CAMERA_AZIMUTH = 135.0
_DEFAULT_CAMERA_ELEVATION = -25.0
_DEFAULT_GEOMETRY_COLOR = (0.78, 0.80, 0.84, 1.0)
_DEFAULT_CAMERA_LENS_MM = 28.0
_DEFAULT_WORLD_COLOR = (0.92, 0.93, 0.96, 1.0)
_DEFAULT_WORLD_STRENGTH = 0.12
_DEFAULT_SUN_ENERGY = 0.35
_SUPPORTED_JOINT_TYPES = {"continuous", "fixed", "prismatic", "revolute"}
_ACTUATED_JOINT_TYPES = {"continuous", "prismatic", "revolute"}
_RIG_METADATA_VERSION = "3"
_RIG_METADATA_VERSION_KEY = "ei_vo_rig_version"
_RIG_METADATA_ROOT_LINKS_KEY = "ei_vo_root_links"
_RIG_METADATA_JOINTS_KEY = "ei_vo_joint_specs"
_RIG_METADATA_ACTUATED_KEY = "ei_vo_actuated_joint_names"


@dataclass(slots=True)
class GeometrySpec:
    kind: str
    size: tuple[float, ...] | None = None
    radius: float | None = None
    length: float | None = None
    mesh_path: str | None = None
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0)


@dataclass(slots=True)
class VisualSpec:
    name: str
    origin_xyz: tuple[float, float, float]
    origin_rpy: tuple[float, float, float]
    geometry: GeometrySpec
    color: tuple[float, float, float, float] | None = None


@dataclass(slots=True)
class LinkSpec:
    name: str
    visuals: list[VisualSpec] = field(default_factory=list)


@dataclass(slots=True)
class JointSpec:
    name: str
    joint_type: str
    parent: str
    child: str
    origin_xyz: tuple[float, float, float]
    origin_rpy: tuple[float, float, float]
    axis: tuple[float, float, float]


@dataclass(slots=True)
class SceneRig:
    link_objects: dict[str, object]
    child_joints: dict[str, list[JointSpec]]
    root_links: list[str]
    actuated_joints: list[JointSpec]


def _parse_args() -> pathlib.Path:
    argv = list(sys.argv)
    if "--" not in argv:
        raise SystemExit("Expected a manifest path after '--'.")
    manifest_args = argv[argv.index("--") + 1 :]
    if len(manifest_args) != 1:
        raise SystemExit("Expected exactly one manifest path.")
    return pathlib.Path(manifest_args[0]).expanduser().resolve(strict=False)


def load_manifest(path: str | pathlib.Path) -> dict[str, Any]:
    manifest_path = pathlib.Path(path).expanduser().resolve(strict=False)
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _parse_floats(
    value: str | None,
    *,
    count: int,
    default: tuple[float, ...],
) -> tuple[float, ...]:
    if value is None:
        return default
    parts = tuple(float(part) for part in value.split())
    if len(parts) != count:
        raise RuntimeError(f"Expected {count} values, got {len(parts)} from {value!r}.")
    return parts


def _parse_rgba(value: str | None) -> tuple[float, float, float, float] | None:
    if value is None:
        return None
    rgba = _parse_floats(value, count=4, default=_DEFAULT_GEOMETRY_COLOR)
    return tuple(max(0.0, min(1.0, float(component))) for component in rgba)


def _joint_sort_key(name: str) -> int:
    import re

    match = re.search(r"(\d+)$", name) or re.search(r"joint[_-]?(\d+)", name)
    return int(match.group(1)) if match else 999


def _rotation_matrix_from_rpy(rpy: tuple[float, float, float]) -> Matrix:
    roll, pitch, yaw = rpy
    cos_roll = math.cos(roll)
    sin_roll = math.sin(roll)
    cos_pitch = math.cos(pitch)
    sin_pitch = math.sin(pitch)
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    return Matrix(
        (
            (
                cos_yaw * cos_pitch,
                cos_yaw * sin_pitch * sin_roll - sin_yaw * cos_roll,
                cos_yaw * sin_pitch * cos_roll + sin_yaw * sin_roll,
            ),
            (
                sin_yaw * cos_pitch,
                sin_yaw * sin_pitch * sin_roll + cos_yaw * cos_roll,
                sin_yaw * sin_pitch * cos_roll - cos_yaw * sin_roll,
            ),
            (
                -sin_pitch,
                cos_pitch * sin_roll,
                cos_pitch * cos_roll,
            ),
        )
    )


def _origin_matrix(
    xyz: tuple[float, float, float],
    rpy: tuple[float, float, float],
) -> Matrix:
    transform = _rotation_matrix_from_rpy(rpy).to_4x4()
    transform.translation = Vector(xyz)
    return transform


def _motion_matrix(joint: JointSpec, value: float) -> Matrix:
    axis = Vector(joint.axis)
    if axis.length_squared <= 1e-12:
        axis = Vector((1.0, 0.0, 0.0))
    else:
        axis.normalize()

    if joint.joint_type == "prismatic":
        transform = Matrix.Identity(4)
        transform.translation = axis * float(value)
        return transform
    if joint.joint_type in {"continuous", "revolute"}:
        return Matrix.Rotation(float(value), 4, axis)
    return Matrix.Identity(4)


def _resolve_mesh_path(model_dir: pathlib.Path, filename: str) -> pathlib.Path | None:
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
        for root in (model_dir, *model_dir.parents):
            for candidate in (root / tail, root / package_name / tail):
                if candidate.is_file():
                    return candidate
        return None

    return None


def _parse_materials(root: ET.Element) -> dict[str, tuple[float, float, float, float]]:
    materials: dict[str, tuple[float, float, float, float]] = {}
    for material in root.findall("material"):
        name = (material.get("name") or "").strip()
        color = material.find("color")
        rgba = _parse_rgba(None if color is None else color.get("rgba"))
        if name and rgba is not None:
            materials[name] = rgba
    return materials


def _parse_geometry(
    geometry: ET.Element,
    *,
    model_dir: pathlib.Path,
) -> GeometrySpec:
    box = geometry.find("box")
    if box is not None:
        size = _parse_floats(box.get("size"), count=3, default=(1.0, 1.0, 1.0))
        return GeometrySpec(kind="box", size=size)

    cylinder = geometry.find("cylinder")
    if cylinder is not None:
        return GeometrySpec(
            kind="cylinder",
            radius=float(cylinder.get("radius", "0.05")),
            length=float(cylinder.get("length", "0.1")),
        )

    sphere = geometry.find("sphere")
    if sphere is not None:
        return GeometrySpec(kind="sphere", radius=float(sphere.get("radius", "0.05")))

    mesh = geometry.find("mesh")
    if mesh is not None:
        filename = mesh.get("filename")
        if not filename:
            raise RuntimeError("URDF mesh geometry is missing its filename attribute.")
        mesh_path = _resolve_mesh_path(model_dir, filename)
        if mesh_path is None:
            raise RuntimeError(f"Could not resolve URDF mesh asset {filename!r}.")
        scale = _parse_floats(mesh.get("scale"), count=3, default=(1.0, 1.0, 1.0))
        return GeometrySpec(kind="mesh", mesh_path=mesh_path.as_posix(), scale=scale)

    raise RuntimeError("Unsupported URDF visual geometry. Supported: box, cylinder, sphere, mesh.")


def _parse_visual(
    visual: ET.Element,
    *,
    link_name: str,
    index: int,
    model_dir: pathlib.Path,
    materials: dict[str, tuple[float, float, float, float]],
) -> VisualSpec | None:
    geometry = visual.find("geometry")
    if geometry is None:
        return None

    origin = visual.find("origin")
    material = visual.find("material")
    inline_color = None if material is None else material.find("color")
    material_name = None if material is None else (material.get("name") or "").strip()
    rgba = _parse_rgba(None if inline_color is None else inline_color.get("rgba"))
    if rgba is None and material_name:
        rgba = materials.get(material_name)

    return VisualSpec(
        name=(visual.get("name") or f"{link_name}_visual_{index}").strip() or f"{link_name}_visual_{index}",
        origin_xyz=_parse_floats(None if origin is None else origin.get("xyz"), count=3, default=(0.0, 0.0, 0.0)),
        origin_rpy=_parse_floats(None if origin is None else origin.get("rpy"), count=3, default=(0.0, 0.0, 0.0)),
        geometry=_parse_geometry(geometry, model_dir=model_dir),
        color=rgba,
    )


def _parse_robot(
    model_path: pathlib.Path,
) -> tuple[dict[str, LinkSpec], dict[str, list[JointSpec]], list[str], list[JointSpec]]:
    tree = ET.parse(model_path)
    root = tree.getroot()
    model_dir = model_path.parent
    materials = _parse_materials(root)

    links: dict[str, LinkSpec] = {}
    for link in root.findall("link"):
        name = (link.get("name") or "").strip()
        if not name:
            continue
        visuals = []
        for index, visual in enumerate(link.findall("visual")):
            parsed_visual = _parse_visual(
                visual,
                link_name=name,
                index=index,
                model_dir=model_dir,
                materials=materials,
            )
            if parsed_visual is not None:
                visuals.append(parsed_visual)
        links[name] = LinkSpec(name=name, visuals=visuals)

    child_joints: dict[str, list[JointSpec]] = {}
    child_links: set[str] = set()
    actuated_joints: list[JointSpec] = []
    for joint in root.findall("joint"):
        joint_type = (joint.get("type") or "").strip().lower()
        if joint_type not in _SUPPORTED_JOINT_TYPES:
            raise RuntimeError(f"Unsupported URDF joint type {joint_type!r}.")

        name = (joint.get("name") or "").strip()
        parent = joint.find("parent")
        child = joint.find("child")
        if not name or parent is None or child is None:
            raise RuntimeError("URDF joint definitions must include name, parent, and child.")
        parent_link = (parent.get("link") or "").strip()
        child_link = (child.get("link") or "").strip()
        if not parent_link or not child_link:
            raise RuntimeError(f"Joint {name!r} is missing parent/child link names.")

        origin = joint.find("origin")
        axis = joint.find("axis")
        joint_spec = JointSpec(
            name=name,
            joint_type=joint_type,
            parent=parent_link,
            child=child_link,
            origin_xyz=_parse_floats(None if origin is None else origin.get("xyz"), count=3, default=(0.0, 0.0, 0.0)),
            origin_rpy=_parse_floats(None if origin is None else origin.get("rpy"), count=3, default=(0.0, 0.0, 0.0)),
            axis=_parse_floats(None if axis is None else axis.get("xyz"), count=3, default=(1.0, 0.0, 0.0)),
        )
        child_joints.setdefault(parent_link, []).append(joint_spec)
        child_links.add(child_link)

        lowered_name = name.lower()
        if joint_type in _ACTUATED_JOINT_TYPES and "finger" not in lowered_name and "gripper" not in lowered_name:
            actuated_joints.append(joint_spec)

        links.setdefault(parent_link, LinkSpec(name=parent_link))
        links.setdefault(child_link, LinkSpec(name=child_link))

    root_links = sorted(link_name for link_name in links if link_name not in child_links)
    if not root_links and links:
        root_links = [sorted(links)[0]]

    actuated_joints.sort(key=lambda joint: _joint_sort_key(joint.name))
    return links, child_joints, root_links, actuated_joints


def _reset_scene() -> bpy.types.Scene:
    scene = bpy.context.scene
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for collection in tuple(bpy.data.collections):
        if collection.users == 0:
            bpy.data.collections.remove(collection)
    for mesh in tuple(bpy.data.meshes):
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)
    for material in tuple(bpy.data.materials):
        if material.users == 0:
            bpy.data.materials.remove(material)
    for light in tuple(bpy.data.lights):
        if light.users == 0:
            bpy.data.lights.remove(light)
    for camera in tuple(bpy.data.cameras):
        if camera.users == 0:
            bpy.data.cameras.remove(camera)
    return scene


def _resolve_scene_cache_path(cache_spec: dict[str, object] | None) -> pathlib.Path | None:
    if cache_spec is None:
        return None
    cache_path = str(cache_spec.get("path") or "").strip()
    if not cache_path:
        return None
    return pathlib.Path(cache_path).expanduser().resolve(strict=False)


def _material_cache() -> dict[tuple[float, float, float, float], bpy.types.Material]:
    return {}


def _set_principled_input(
    principled: Any,
    names: tuple[str, ...],
    value: float,
) -> None:
    for name in names:
        socket = principled.inputs.get(name)
        if socket is not None:
            socket.default_value = value
            return


def _get_material(
    cache: dict[tuple[float, float, float, float], bpy.types.Material],
    rgba: tuple[float, float, float, float] | None,
) -> bpy.types.Material:
    resolved_rgba = rgba or _DEFAULT_GEOMETRY_COLOR
    material = cache.get(resolved_rgba)
    if material is not None:
        return material

    material = bpy.data.materials.new(
        name="ei_vo_"
        + "_".join(f"{int(round(component * 255.0)):03d}" for component in resolved_rgba)
    )
    material.use_nodes = True
    principled = material.node_tree.nodes.get("Principled BSDF")
    if principled is not None:
        principled.inputs["Base Color"].default_value = resolved_rgba
        _set_principled_input(principled, ("Roughness",), 0.78)
        _set_principled_input(principled, ("Metallic",), 0.0)
        _set_principled_input(principled, ("Specular IOR Level", "Specular"), 0.18)
        if "Alpha" in principled.inputs:
            principled.inputs["Alpha"].default_value = resolved_rgba[3]
    if hasattr(material, "blend_method") and resolved_rgba[3] < 0.999:
        material.blend_method = "BLEND"
    if hasattr(material, "shadow_method") and resolved_rgba[3] < 0.999:
        material.shadow_method = "HASHED"
    cache[resolved_rgba] = material
    return material


def _assign_material(
    obj: bpy.types.Object,
    cache: dict[tuple[float, float, float, float], bpy.types.Material],
    rgba: tuple[float, float, float, float] | None,
) -> None:
    if obj.type != "MESH":
        return
    material = _get_material(cache, rgba)
    if obj.data.materials:
        for index in range(len(obj.data.materials)):
            obj.data.materials[index] = material
    else:
        obj.data.materials.append(material)


def _parent_as_local(child: bpy.types.Object, parent: bpy.types.Object) -> None:
    bpy.context.view_layer.update()
    local_matrix = child.matrix_world.copy()
    child.parent = parent
    child.matrix_parent_inverse = Matrix.Identity(4)
    child.matrix_basis = local_matrix


def _create_empty(name: str, scene: bpy.types.Scene) -> bpy.types.Object:
    empty = bpy.data.objects.new(name, None)
    empty.empty_display_type = "PLAIN_AXES"
    empty.empty_display_size = 0.03
    scene.collection.objects.link(empty)
    return empty


def _newly_imported_objects(callback) -> list[bpy.types.Object]:
    before = set(bpy.data.objects.keys())
    callback()
    return [bpy.data.objects[name] for name in bpy.data.objects.keys() if name not in before]


def _import_mesh_objects(mesh_path: pathlib.Path) -> list[bpy.types.Object]:
    suffix = mesh_path.suffix.lower()
    if suffix == ".stl":
        if hasattr(bpy.ops.wm, "stl_import"):
            imported = _newly_imported_objects(lambda: bpy.ops.wm.stl_import(filepath=mesh_path.as_posix()))
        else:
            imported = _newly_imported_objects(
                lambda: bpy.ops.import_mesh.stl(filepath=mesh_path.as_posix())
            )
        return imported
    if suffix == ".obj":
        if hasattr(bpy.ops.wm, "obj_import"):
            imported = _newly_imported_objects(lambda: bpy.ops.wm.obj_import(filepath=mesh_path.as_posix()))
        else:
            imported = _newly_imported_objects(
                lambda: bpy.ops.import_scene.obj(filepath=mesh_path.as_posix())
            )
        return imported
    if suffix == ".dae":
        return _newly_imported_objects(lambda: bpy.ops.wm.collada_import(filepath=mesh_path.as_posix()))
    if suffix in {".glb", ".gltf"}:
        return _newly_imported_objects(lambda: bpy.ops.import_scene.gltf(filepath=mesh_path.as_posix()))
    raise RuntimeError(f"Unsupported mesh format {mesh_path.suffix!r} for Blender import.")


def _create_visual_geometry(
    scene: bpy.types.Scene,
    visual_root: bpy.types.Object,
    visual: VisualSpec,
    cache: dict[tuple[float, float, float, float], bpy.types.Material],
) -> None:
    geometry = visual.geometry
    if geometry.kind == "box":
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, 0.0, 0.0))
        obj = bpy.context.active_object
        obj.scale = Vector(geometry.size or (1.0, 1.0, 1.0))
        _parent_as_local(obj, visual_root)
        _assign_material(obj, cache, visual.color)
        return

    if geometry.kind == "cylinder":
        bpy.ops.mesh.primitive_cylinder_add(
            vertices=48,
            radius=float(geometry.radius or 0.05),
            depth=float(geometry.length or 0.1),
            location=(0.0, 0.0, 0.0),
        )
        obj = bpy.context.active_object
        _parent_as_local(obj, visual_root)
        _assign_material(obj, cache, visual.color)
        return

    if geometry.kind == "sphere":
        bpy.ops.mesh.primitive_uv_sphere_add(
            segments=48,
            ring_count=24,
            radius=float(geometry.radius or 0.05),
            location=(0.0, 0.0, 0.0),
        )
        obj = bpy.context.active_object
        _parent_as_local(obj, visual_root)
        _assign_material(obj, cache, visual.color)
        return

    if geometry.kind == "mesh":
        mesh_path = pathlib.Path(geometry.mesh_path or "")
        imported = _import_mesh_objects(mesh_path)
        if not imported:
            raise RuntimeError(f"Importing {mesh_path.name!r} did not create any Blender objects.")
        imported_names = {obj.name for obj in imported}
        for obj in imported:
            if obj.parent is None or obj.parent.name not in imported_names:
                _parent_as_local(obj, visual_root)
        for obj in imported:
            _assign_material(obj, cache, visual.color)
        return

    raise RuntimeError(f"Unsupported geometry kind {geometry.kind!r}.")


def _build_robot(
    scene: bpy.types.Scene,
    links: dict[str, LinkSpec],
    child_joints: dict[str, list[JointSpec]],
    root_links: list[str],
    actuated_joints: list[JointSpec],
) -> SceneRig:
    cache = _material_cache()
    link_objects: dict[str, bpy.types.Object] = {}
    for link_name, link in links.items():
        link_root = _create_empty(f"link::{link_name}", scene)
        link_objects[link_name] = link_root
        for index, visual in enumerate(link.visuals):
            visual_root = _create_empty(f"visual::{link_name}::{index}", scene)
            visual_root.parent = link_root
            visual_root.matrix_parent_inverse = Matrix.Identity(4)
            visual_root.matrix_basis = _origin_matrix(visual.origin_xyz, visual.origin_rpy)
            visual_root.scale = Vector(visual.geometry.scale)
            _create_visual_geometry(scene, visual_root, visual, cache)
    return SceneRig(
        link_objects=link_objects,
        child_joints=child_joints,
        root_links=root_links,
        actuated_joints=actuated_joints,
    )


def _joint_spec_to_dict(joint: JointSpec) -> dict[str, object]:
    return {
        "name": joint.name,
        "joint_type": joint.joint_type,
        "parent": joint.parent,
        "child": joint.child,
        "origin_xyz": list(joint.origin_xyz),
        "origin_rpy": list(joint.origin_rpy),
        "axis": list(joint.axis),
    }


def _joint_spec_from_dict(data: dict[str, object]) -> JointSpec:
    return JointSpec(
        name=str(data["name"]),
        joint_type=str(data["joint_type"]),
        parent=str(data["parent"]),
        child=str(data["child"]),
        origin_xyz=tuple(float(value) for value in data["origin_xyz"]),
        origin_rpy=tuple(float(value) for value in data["origin_rpy"]),
        axis=tuple(float(value) for value in data["axis"]),
    )


def _store_rig_metadata(scene: bpy.types.Scene, rig: SceneRig) -> None:
    joints = [
        _joint_spec_to_dict(joint)
        for parent_name in sorted(rig.child_joints)
        for joint in rig.child_joints[parent_name]
    ]
    scene[_RIG_METADATA_VERSION_KEY] = _RIG_METADATA_VERSION
    scene[_RIG_METADATA_ROOT_LINKS_KEY] = json.dumps(rig.root_links, separators=(",", ":"))
    scene[_RIG_METADATA_JOINTS_KEY] = json.dumps(joints, separators=(",", ":"))
    scene[_RIG_METADATA_ACTUATED_KEY] = json.dumps(
        [joint.name for joint in rig.actuated_joints],
        separators=(",", ":"),
    )


def _load_cached_rig(scene: bpy.types.Scene) -> SceneRig | None:
    if str(scene.get(_RIG_METADATA_VERSION_KEY, "")) != _RIG_METADATA_VERSION:
        return None

    try:
        root_links = [str(name) for name in json.loads(str(scene[_RIG_METADATA_ROOT_LINKS_KEY]))]
        joints_data = json.loads(str(scene[_RIG_METADATA_JOINTS_KEY]))
        actuated_names = [str(name) for name in json.loads(str(scene[_RIG_METADATA_ACTUATED_KEY]))]
    except Exception:
        return None

    link_objects: dict[str, bpy.types.Object] = {}
    for obj in scene.objects:
        if obj.name.startswith("link::"):
            link_objects[obj.name.split("::", 1)[1]] = obj
    if not link_objects:
        return None

    child_joints: dict[str, list[JointSpec]] = {}
    joints_by_name: dict[str, JointSpec] = {}
    try:
        for item in joints_data:
            joint = _joint_spec_from_dict(item)
            if joint.parent not in link_objects or joint.child not in link_objects:
                return None
            child_joints.setdefault(joint.parent, []).append(joint)
            joints_by_name[joint.name] = joint
    except Exception:
        return None

    for root_link in root_links:
        if root_link not in link_objects:
            return None

    actuated_joints: list[JointSpec] = []
    for joint_name in actuated_names:
        joint = joints_by_name.get(joint_name)
        if joint is None:
            return None
        actuated_joints.append(joint)

    return SceneRig(
        link_objects=link_objects,
        child_joints=child_joints,
        root_links=root_links,
        actuated_joints=actuated_joints,
    )


def _apply_pose(rig: SceneRig, q: list[float]) -> None:
    if len(q) != len(rig.actuated_joints):
        raise RuntimeError(
            f"Trajectory row has {len(q)} values, expected {len(rig.actuated_joints)} joint values."
        )

    joint_values = {joint.name: float(q[index]) for index, joint in enumerate(rig.actuated_joints)}
    resolved: set[str] = set()

    def visit(link_name: str, transform: Matrix) -> None:
        resolved.add(link_name)
        rig.link_objects[link_name].matrix_world = transform
        for joint in rig.child_joints.get(link_name, []):
            child_transform = (
                transform
                @ _origin_matrix(joint.origin_xyz, joint.origin_rpy)
                @ _motion_matrix(joint, joint_values.get(joint.name, 0.0))
            )
            visit(joint.child, child_transform)

    identity = Matrix.Identity(4)
    for root_link in rig.root_links:
        visit(root_link, identity)
    for link_name, link_object in rig.link_objects.items():
        if link_name not in resolved:
            link_object.matrix_world = identity
    bpy.context.view_layer.update()


def _scene_bounds(scene: bpy.types.Scene) -> tuple[Vector, float, float]:
    min_corner, max_corner = _scene_extents(scene)
    center = (min_corner + max_corner) * 0.5
    span = max_corner - min_corner
    radius = max(float(max(span.x, span.y, span.z)), 0.5) * 0.5
    return center, radius, float(min_corner.z)


def _scene_extents(scene: bpy.types.Scene) -> tuple[Vector, Vector]:
    corners: list[Vector] = []
    for obj in scene.objects:
        if obj.type != "MESH":
            continue
        corners.extend(obj.matrix_world @ Vector(corner) for corner in obj.bound_box)

    if not corners:
        min_corner = Vector((-0.25, -0.25, -0.25))
        max_corner = Vector((0.25, 0.25, 0.25))
        return min_corner, max_corner

    min_x = min(corner.x for corner in corners)
    min_y = min(corner.y for corner in corners)
    min_z = min(corner.z for corner in corners)
    max_x = max(corner.x for corner in corners)
    max_y = max(corner.y for corner in corners)
    max_z = max(corner.z for corner in corners)
    return Vector((min_x, min_y, min_z)), Vector((max_x, max_y, max_z))


def _iter_descendant_meshes(root: bpy.types.Object) -> list[bpy.types.Object]:
    meshes: list[bpy.types.Object] = []
    stack = list(root.children)
    while stack:
        obj = stack.pop()
        stack.extend(obj.children)
        if obj.type == "MESH":
            meshes.append(obj)
    return meshes


def _project_world_point(
    scene: bpy.types.Scene,
    camera: bpy.types.Object,
    point: Vector,
) -> dict[str, object]:
    from bpy_extras.object_utils import world_to_camera_view

    view = world_to_camera_view(scene, camera, point)
    width = float(scene.render.resolution_x * scene.render.resolution_percentage) / 100.0
    height = float(scene.render.resolution_y * scene.render.resolution_percentage) / 100.0
    return {
        "normalized": [float(view.x), float(view.y), float(view.z)],
        "pixel": [float(view.x * width), float((1.0 - view.y) * height)],
        "visible": bool(view.z >= 0.0 and 0.0 <= view.x <= 1.0 and 0.0 <= view.y <= 1.0),
    }


def _collect_link_debug(
    scene: bpy.types.Scene,
    rig: SceneRig,
) -> list[dict[str, object]]:
    camera = scene.camera
    if camera is None:
        return []

    link_debug: list[dict[str, object]] = []
    for link_name in sorted(rig.link_objects):
        link_root = rig.link_objects[link_name]
        meshes = _iter_descendant_meshes(link_root)
        link_entry: dict[str, object] = {
            "name": link_name,
            "root_world": [float(value) for value in link_root.matrix_world.translation],
            "root_projection": _project_world_point(scene, camera, link_root.matrix_world.translation),
            "mesh_object_names": [obj.name for obj in meshes],
        }
        if not meshes:
            link_entry["visible"] = False
            link_debug.append(link_entry)
            continue

        corners = [obj.matrix_world @ Vector(corner) for obj in meshes for corner in obj.bound_box]
        min_corner = Vector(
            (
                min(corner.x for corner in corners),
                min(corner.y for corner in corners),
                min(corner.z for corner in corners),
            )
        )
        max_corner = Vector(
            (
                max(corner.x for corner in corners),
                max(corner.y for corner in corners),
                max(corner.z for corner in corners),
            )
        )
        center = (min_corner + max_corner) * 0.5
        projected_corners = [_project_world_point(scene, camera, corner) for corner in corners]
        pixel_points = [item["pixel"] for item in projected_corners]
        link_entry.update(
            {
                "visual_center_world": [float(value) for value in center],
                "visual_center_projection": _project_world_point(scene, camera, center),
                "world_bounds": {
                    "min": [float(value) for value in min_corner],
                    "max": [float(value) for value in max_corner],
                },
                "pixel_bounds": {
                    "min": [
                        float(min(point[0] for point in pixel_points)),
                        float(min(point[1] for point in pixel_points)),
                    ],
                    "max": [
                        float(max(point[0] for point in pixel_points)),
                        float(max(point[1] for point in pixel_points)),
                    ],
                },
                "visible": any(bool(item["visible"]) for item in projected_corners),
            }
        )
        link_debug.append(link_entry)
    return link_debug


def _write_link_debug(
    scene: bpy.types.Scene,
    rig: SceneRig,
    *,
    frame_index: int,
    output_path: pathlib.Path,
) -> None:
    camera = scene.camera
    if camera is None:
        raise RuntimeError("Cannot write Blender link debug output without a camera.")

    debug_payload = {
        "frame_index": int(frame_index),
        "image_size": {
            "width": int(scene.render.resolution_x * scene.render.resolution_percentage / 100),
            "height": int(scene.render.resolution_y * scene.render.resolution_percentage / 100),
        },
        "camera": {
            "name": camera.name,
            "location": [float(value) for value in camera.location],
            "rotation_euler": [float(value) for value in camera.rotation_euler],
            "lens_mm": float(camera.data.lens),
        },
        "links": _collect_link_debug(scene, rig),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(debug_payload, indent=2), encoding="utf-8")
    print(f"[ei-vo:blender] wrote link debug {output_path}")


def _trajectory_scene_bounds(
    scene: bpy.types.Scene,
    rig: SceneRig,
    trajectory: list[list[float]],
) -> tuple[Vector, float, float]:
    min_corner: Vector | None = None
    max_corner: Vector | None = None
    for q in trajectory:
        _apply_pose(rig, q)
        current_min, current_max = _scene_extents(scene)
        if min_corner is None:
            min_corner = current_min.copy()
            max_corner = current_max.copy()
            continue
        min_corner.x = min(min_corner.x, current_min.x)
        min_corner.y = min(min_corner.y, current_min.y)
        min_corner.z = min(min_corner.z, current_min.z)
        max_corner.x = max(max_corner.x, current_max.x)
        max_corner.y = max(max_corner.y, current_max.y)
        max_corner.z = max(max_corner.z, current_max.z)

    assert min_corner is not None
    assert max_corner is not None
    center = (min_corner + max_corner) * 0.5
    span = max_corner - min_corner
    radius = max(float(max(span.x, span.y, span.z)), 0.5) * 0.5
    return center, radius, float(min_corner.z)


def _load_scene_from_cache(cache_path: pathlib.Path) -> tuple[bpy.types.Scene, SceneRig] | None:
    if not cache_path.is_file():
        return None
    try:
        bpy.ops.wm.open_mainfile(filepath=cache_path.as_posix())
    except Exception as exc:
        print(f"[ei-vo:blender] failed to open scene cache {cache_path}: {exc}")
        return None

    scene = bpy.context.scene
    rig = _load_cached_rig(scene)
    if rig is None:
        print(f"[ei-vo:blender] scene cache is missing ei-vo rig metadata: {cache_path}")
        return None
    print(f"[ei-vo:blender] loaded cached scene {cache_path}")
    return scene, rig


def _save_scene_cache(cache_path: pathlib.Path) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        bpy.ops.wm.save_as_mainfile(filepath=cache_path.as_posix())
        print(f"[ei-vo:blender] saved scene cache {cache_path}")
    except Exception as exc:
        print(f"[ei-vo:blender] failed to save scene cache {cache_path}: {exc}")


def _configure_world(scene: bpy.types.Scene) -> None:
    if scene.world is None:
        scene.world = bpy.data.worlds.new("ei_vo_world")
    world = scene.world
    world.use_nodes = True
    background = world.node_tree.nodes.get("Background")
    if background is not None:
        background.inputs[0].default_value = _DEFAULT_WORLD_COLOR
        background.inputs[1].default_value = _DEFAULT_WORLD_STRENGTH


def _add_floor(scene: bpy.types.Scene, *, radius: float, min_z: float) -> None:
    bpy.ops.mesh.primitive_plane_add(
        size=max(radius * 8.0, 2.0),
        location=(0.0, 0.0, min_z),
    )
    floor = bpy.context.active_object
    floor.name = "floor"
    material = bpy.data.materials.new(name="ei_vo_floor")
    material.use_nodes = True
    principled = material.node_tree.nodes.get("Principled BSDF")
    if principled is not None:
        principled.inputs["Base Color"].default_value = (0.93, 0.94, 0.96, 1.0)
        principled.inputs["Roughness"].default_value = 0.75
    floor.data.materials.append(material)


def _add_lights(radius: float, center: Vector) -> None:
    bpy.ops.object.light_add(
        type="SUN",
        location=(center.x + radius * 2.0, center.y - radius * 2.0, center.z + radius * 3.0),
    )
    sun = bpy.context.active_object
    sun.name = "sun_key"
    sun.data.energy = _DEFAULT_SUN_ENERGY
    sun.rotation_euler = (math.radians(45.0), 0.0, math.radians(35.0))

    bpy.ops.object.light_add(
        type="AREA",
        location=(center.x, center.y - radius * 1.5, center.z + radius * 3.0),
    )
    area = bpy.context.active_object
    area.name = "area_fill"
    area.data.energy = max(float(radius) * 220.0, 60.0)
    area.data.shape = "RECTANGLE"
    area.data.size = max(radius * 3.5, 1.4)
    if hasattr(area.data, "size_y"):
        area.data.size_y = max(radius * 2.2, 1.0)


def _resolve_camera_spec(
    camera: dict | None,
    *,
    center: Vector,
    radius: float,
    render_width: int,
    render_height: int,
) -> dict[str, object]:
    scene_radius = max(float(radius), 0.25)
    defaults = {
        "distance": default_blender_camera_distance(
            scene_radius,
            width=int(render_width),
            height=int(render_height),
            lens_mm=_DEFAULT_CAMERA_LENS_MM,
        ),
        "azimuth": _DEFAULT_CAMERA_AZIMUTH,
        "elevation": _DEFAULT_CAMERA_ELEVATION,
        "lookat": [float(center.x), float(center.y), float(center.z + scene_radius * 0.55)],
    }
    if camera is None:
        return defaults
    return {
        "distance": defaults["distance"] if camera.get("distance") is None else float(camera["distance"]),
        "azimuth": defaults["azimuth"] if camera.get("azimuth") is None else float(camera["azimuth"]),
        "elevation": defaults["elevation"] if camera.get("elevation") is None else float(camera["elevation"]),
        "lookat": defaults["lookat"] if camera.get("lookat") is None else [float(value) for value in camera["lookat"]],
    }


def _camera_offset(distance: float, azimuth_deg: float, elevation_deg: float) -> Vector:
    azimuth = math.radians(float(azimuth_deg))
    elevation = math.radians(float(elevation_deg))
    cos_elevation = math.cos(elevation)
    return Vector(
        (
            -float(distance) * cos_elevation * math.cos(azimuth),
            -float(distance) * cos_elevation * math.sin(azimuth),
            -float(distance) * math.sin(elevation),
        )
    )


def _add_camera(scene: bpy.types.Scene, camera_spec: dict[str, object], radius: float) -> None:
    lookat = Vector(camera_spec["lookat"])
    location = lookat + _camera_offset(
        float(camera_spec["distance"]),
        float(camera_spec["azimuth"]),
        float(camera_spec["elevation"]),
    )

    bpy.ops.object.camera_add(location=location)
    camera = bpy.context.active_object
    camera.name = "render_camera"
    camera.data.lens = _DEFAULT_CAMERA_LENS_MM
    camera.data.clip_start = 0.01
    camera.data.clip_end = max(radius * 40.0, 80.0)
    direction = lookat - location
    camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    scene.camera = camera


def _resolve_engine(scene: bpy.types.Scene, requested: str) -> str:
    candidate_map = {
        "cycles": ("CYCLES",),
        "eevee": ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE"),
        "workbench": ("BLENDER_WORKBENCH",),
    }
    if requested not in candidate_map:
        raise RuntimeError(f"Unsupported Blender engine option {requested!r}.")
    for candidate in candidate_map[requested]:
        try:
            scene.render.engine = candidate
            return candidate
        except Exception:
            continue
    raise RuntimeError(f"Requested Blender render engine {requested!r} is not available in this Blender build.")


def _configure_render(scene: bpy.types.Scene, render_spec: dict[str, object], *, image_format: str) -> None:
    scene.render.resolution_x = int(render_spec["width"])
    scene.render.resolution_y = int(render_spec["height"])
    scene.render.resolution_percentage = 100
    scene.render.use_file_extension = True
    scene.render.film_transparent = False
    scene.render.image_settings.file_format = image_format
    scene.render.image_settings.color_mode = "RGB"
    if image_format == "JPEG":
        scene.render.image_settings.quality = 95

    resolved_engine = _resolve_engine(scene, str(render_spec["engine"]))
    samples = int(render_spec["samples"])
    if resolved_engine == "CYCLES" and hasattr(scene, "cycles"):
        scene.cycles.samples = samples
    elif hasattr(scene, "eevee"):
        scene.eevee.taa_render_samples = samples


def render_manifest(manifest: dict[str, Any]) -> None:
    trajectory = manifest["trajectory"]
    if not trajectory:
        raise RuntimeError("Trajectory must contain at least one frame.")

    model_path = pathlib.Path(manifest["model_path"]).expanduser().resolve(strict=False)
    scene_cache_path = _resolve_scene_cache_path(manifest.get("scene_cache"))
    cached_scene = None if scene_cache_path is None else _load_scene_from_cache(scene_cache_path)
    if cached_scene is None:
        scene = _reset_scene()
        _configure_world(scene)

        links, child_joints, root_links, actuated_joints = _parse_robot(model_path)
        if len(trajectory[0]) != len(actuated_joints):
            raise RuntimeError(
                f"Trajectory DOF ({len(trajectory[0])}) does not match URDF actuated joints ({len(actuated_joints)})."
            )

        rig = _build_robot(scene, links, child_joints, root_links, actuated_joints)
        _store_rig_metadata(scene, rig)
        if scene_cache_path is not None:
            _save_scene_cache(scene_cache_path)
    else:
        scene, rig = cached_scene
        _configure_world(scene)
        if len(trajectory[0]) != len(rig.actuated_joints):
            raise RuntimeError(
                f"Trajectory DOF ({len(trajectory[0])}) does not match cached rig actuated joints ({len(rig.actuated_joints)})."
            )

    center, radius, min_z = _trajectory_scene_bounds(scene, rig, trajectory)
    _apply_pose(rig, trajectory[0])
    if bool(manifest["render"].get("floor", True)):
        _add_floor(scene, radius=radius, min_z=min_z)
    _add_lights(radius, center)
    camera_spec = _resolve_camera_spec(
        manifest.get("camera"),
        center=center,
        radius=radius,
        render_width=int(manifest["render"]["width"]),
        render_height=int(manifest["render"]["height"]),
    )
    _add_camera(scene, camera_spec, radius)

    output = manifest["output"]
    if output["kind"] == "video":
        _configure_render(scene, manifest["render"], image_format="PNG")
        frame_dir = pathlib.Path(output["frame_dir"]).expanduser().resolve(strict=False)
        frame_dir.mkdir(parents=True, exist_ok=True)
        for index, q in enumerate(trajectory):
            _apply_pose(rig, q)
            bpy.context.view_layer.update()
            scene.render.filepath = (frame_dir / f"{index:07d}.png").as_posix()
            bpy.ops.render.render(write_still=True)
        return

    if output["kind"] == "image":
        _configure_render(scene, manifest["render"], image_format=str(output["format"]))
        frame_index = max(0, min(int(output["frame_index"]), len(trajectory) - 1))
        _apply_pose(rig, trajectory[frame_index])
        bpy.context.view_layer.update()
        debug = manifest.get("debug")
        if isinstance(debug, dict) and debug.get("links_path"):
            _write_link_debug(
                scene,
                rig,
                frame_index=frame_index,
                output_path=pathlib.Path(str(debug["links_path"])).expanduser().resolve(strict=False),
            )
        scene.render.filepath = pathlib.Path(output["path"]).as_posix()
        bpy.ops.render.render(write_still=True)
        return

    raise RuntimeError(f"Unknown Blender output kind {output['kind']!r}.")


def main() -> None:
    manifest_path = _parse_args()
    render_manifest(load_manifest(manifest_path))


if __name__ == "__main__":
    main()
