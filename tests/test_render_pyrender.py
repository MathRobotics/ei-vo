import importlib
import pathlib
import sys
import types

import numpy as np
import pytest

from ei_vo.core import RobotModel, Trajectory


def test_urdfpy_compat_shims_restore_numpy_removed_aliases(monkeypatch):
    compat = importlib.import_module("ei_vo.render._urdfpy")

    for name in ("infty", "Infinity", "Inf", "float", "float_", "int", "bool"):
        monkeypatch.delitem(compat.np.__dict__, name, raising=False)

    compat.install_urdfpy_compat_shims()

    assert compat.np.infty == compat.np.inf
    assert compat.np.Infinity == compat.np.inf
    assert compat.np.Inf == compat.np.inf
    assert compat.np.float is float
    assert compat.np.float_ is compat.np.float64
    assert compat.np.int is int
    assert compat.np.bool is bool


def test_pyrender_auto_selects_egl_for_forwarded_display(monkeypatch):
    sys.modules.pop("ei_vo.render.render_pyrender", None)
    pyrender_renderer = importlib.import_module("ei_vo.render.render_pyrender")
    monkeypatch.setenv("DISPLAY", "localhost:10.0")
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.delenv("PYOPENGL_PLATFORM", raising=False)
    monkeypatch.setenv("SSH_CONNECTION", "a b c d")

    runtime = pyrender_renderer._prepare_pyopengl_platform()

    assert runtime == {"platform": "egl", "auto_selected": True}
    assert pyrender_renderer.os.environ["PYOPENGL_PLATFORM"] == "egl"


def test_pyrender_prefers_default_egl_display_without_explicit_device(monkeypatch):
    sys.modules.pop("ei_vo.render.render_pyrender", None)
    pyrender_renderer = importlib.import_module("ei_vo.render.render_pyrender")
    egl_module = types.SimpleNamespace()
    calls = []

    class DummyDevice:
        def __init__(self, handle):
            self.handle = handle

    egl_module.EGLDevice = DummyDevice
    egl_module.get_device_by_index = lambda device_id: calls.append(device_id) or DummyDevice(device_id)
    pyrender = types.SimpleNamespace(platforms=types.SimpleNamespace(egl=egl_module))

    with pyrender_renderer._prefer_default_egl_display(pyrender, {"platform": "egl", "auto_selected": True}):
        device = egl_module.get_device_by_index(7)

    assert calls == []
    assert isinstance(device, DummyDevice)
    assert device.handle is None

    restored = egl_module.get_device_by_index(3)
    assert calls == [3]
    assert restored.handle == 3


def test_pyrender_keeps_explicit_egl_device(monkeypatch):
    sys.modules.pop("ei_vo.render.render_pyrender", None)
    pyrender_renderer = importlib.import_module("ei_vo.render.render_pyrender")
    monkeypatch.setenv("EGL_DEVICE_ID", "2")
    egl_module = types.SimpleNamespace()
    calls = []

    class DummyDevice:
        def __init__(self, handle):
            self.handle = handle

    egl_module.EGLDevice = DummyDevice
    egl_module.get_device_by_index = lambda device_id: calls.append(device_id) or DummyDevice(device_id)
    pyrender = types.SimpleNamespace(platforms=types.SimpleNamespace(egl=egl_module))

    with pyrender_renderer._prefer_default_egl_display(pyrender, {"platform": "egl", "auto_selected": False}):
        device = egl_module.get_device_by_index(2)

    assert calls == [2]
    assert device.handle == 2


def _install_dummy_pyrender_runtime(monkeypatch):
    captured = {
        "mesh_inputs": [],
        "adds": [],
        "set_pose": [],
        "renders": [],
        "renderer_deleted": False,
        "viewer_calls": [],
        "viewer_lock_acquires": 0,
        "viewer_lock_releases": 0,
    }

    class DummyNode:
        def __init__(self, obj, pose):
            self.obj = obj
            self.pose = np.asarray(pose, dtype=float)

    class DummyScene:
        def __init__(self, bg_color=None, ambient_light=None, name=None):
            self.bg_color = None if bg_color is None else np.asarray(bg_color, dtype=float)
            self.ambient_light = None if ambient_light is None else np.asarray(ambient_light, dtype=float)
            self.name = name
            self.main_camera_node = None

        def add(self, obj, pose=None, name=None, parent_node=None, parent_name=None):
            del name, parent_node, parent_name
            node = DummyNode(obj, np.eye(4, dtype=float) if pose is None else pose)
            captured["adds"].append({"obj": obj, "pose": node.pose.copy()})
            return node

        def set_pose(self, node, pose):
            node.pose = np.asarray(pose, dtype=float)
            captured["set_pose"].append(node.pose.copy())

    class DummyMesh:
        def __init__(self, source):
            self.source = source

        @staticmethod
        def from_trimesh(mesh, smooth=True):
            captured["mesh_inputs"].append({"mesh": mesh, "smooth": smooth})
            return DummyMesh(mesh)

    class DummyPerspectiveCamera:
        def __init__(self, yfov, aspectRatio, znear, zfar):
            self.yfov = yfov
            self.aspectRatio = aspectRatio
            self.znear = znear
            self.zfar = zfar

    class DummyDirectionalLight:
        def __init__(self, color, intensity):
            self.color = np.asarray(color, dtype=float)
            self.intensity = float(intensity)

    class DummyOffscreenRenderer:
        def __init__(self, viewport_width, viewport_height):
            self.viewport_width = viewport_width
            self.viewport_height = viewport_height

        def render(self, scene):
            del scene
            frame = np.full((self.viewport_height, self.viewport_width, 3), 64, dtype=np.uint8)
            captured["renders"].append(frame.copy())
            return frame, np.zeros((self.viewport_height, self.viewport_width), dtype=np.float32)

        def delete(self):
            captured["renderer_deleted"] = True

    class DummyLock:
        def acquire(self):
            captured["viewer_lock_acquires"] += 1

        def release(self):
            captured["viewer_lock_releases"] += 1

    class DummyViewer:
        def __init__(
            self,
            scene,
            viewport_size=None,
            render_flags=None,
            viewer_flags=None,
            registered_keys=None,
            run_in_thread=False,
            **kwargs,
        ):
            del render_flags, registered_keys
            self.scene = scene
            self.viewport_size = viewport_size
            self.run_in_thread = run_in_thread
            self.viewer_flags = dict(viewer_flags or {})
            self.viewer_flags.update(kwargs)
            self.render_lock = DummyLock()
            self.is_active = True
            self._camera_node = types.SimpleNamespace(
                matrix=np.asarray(scene.main_camera_node.pose, dtype=float).copy(),
                camera=scene.main_camera_node.obj,
            )
            self._trackball = types.SimpleNamespace(
                _n_target=np.asarray(self.viewer_flags.get("view_center", np.zeros(3)), dtype=float).copy()
            )
            captured["viewer_instance"] = self
            captured["viewer_calls"].append(
                {
                    "viewport_size": viewport_size,
                    "run_in_thread": run_in_thread,
                    "viewer_flags": dict(self.viewer_flags),
                }
            )

        def close_external(self):
            self.is_active = False

    pyrender = types.ModuleType("pyrender")
    pyrender.Scene = DummyScene
    pyrender.Mesh = DummyMesh
    pyrender.PerspectiveCamera = DummyPerspectiveCamera
    pyrender.DirectionalLight = DummyDirectionalLight
    pyrender.OffscreenRenderer = DummyOffscreenRenderer
    pyrender.Viewer = DummyViewer

    trimesh = types.ModuleType("trimesh")
    trimesh.creation = types.SimpleNamespace(
        box=lambda extents: DummyMeshGeometry("box", [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]]),
        cylinder=lambda radius, height: DummyMeshGeometry(
            "cylinder",
            [[-radius, -radius, -0.5 * height], [radius, radius, 0.5 * height]],
        ),
        icosphere=lambda radius: DummyMeshGeometry(
            "sphere",
            [[-radius, -radius, -radius], [radius, radius, radius]],
        ),
    )

    class DummyMeshGeometry:
        def __init__(self, name, bounds):
            self.name = name
            self.bounds = np.asarray(bounds, dtype=float)

        def copy(self):
            return DummyMeshGeometry(self.name, self.bounds.copy())

    class DummyMaterial:
        def __init__(self, rgba):
            self.color = np.asarray(rgba, dtype=float)

    class DummyGeometry:
        def __init__(self, *, meshes=None):
            self.box = None
            self.cylinder = None
            self.sphere = None
            self.mesh = types.SimpleNamespace(meshes=list(meshes or ()), scale=None) if meshes is not None else None

    class DummyVisual:
        def __init__(self, geometry, origin=None, material=None):
            self.geometry = geometry
            self.origin = np.eye(4, dtype=float) if origin is None else np.asarray(origin, dtype=float)
            self.material = material

    class DummyLink:
        def __init__(self, name, visuals):
            self.name = name
            self.visuals = list(visuals)

    class DummyURDFRobot:
        def __init__(self):
            self.mesh_a = DummyMeshGeometry("base", [[-0.2, -0.1, -0.1], [0.2, 0.1, 0.1]])
            self.mesh_b = DummyMeshGeometry("arm", [[-0.05, -0.05, -0.4], [0.05, 0.05, 0.4]])
            self.base = DummyLink(
                "base",
                [DummyVisual(DummyGeometry(meshes=[self.mesh_a]), material=DummyMaterial([0.8, 0.2, 0.2, 1.0]))],
            )
            self.arm = DummyLink(
                "arm",
                [DummyVisual(DummyGeometry(meshes=[self.mesh_b]), material=DummyMaterial([0.2, 0.4, 0.8, 1.0]))],
            )
            self.links = [self.base, self.arm]

        def link_fk(self, cfg=None):
            cfg = dict(cfg or {})
            joint1 = float(cfg.get("joint1", 0.0))
            joint2 = float(cfg.get("joint2", 0.0))
            pose_a = np.eye(4, dtype=float)
            pose_a[0, 3] = joint1
            pose_b = np.eye(4, dtype=float)
            pose_b[1, 3] = joint2
            return {
                self.base: pose_a,
                self.arm: pose_b,
            }

    class DummyURDF:
        @staticmethod
        def load(path):
            captured["loaded_path"] = path
            return DummyURDFRobot()

    urdfpy = types.ModuleType("urdfpy")
    urdfpy.URDF = DummyURDF

    monkeypatch.setitem(sys.modules, "pyrender", pyrender)
    monkeypatch.setitem(sys.modules, "trimesh", trimesh)
    monkeypatch.setitem(sys.modules, "urdfpy", urdfpy)
    return captured


def test_pyrender_renderer_requires_record_path(monkeypatch, tmp_path):
    captured = _install_dummy_pyrender_runtime(monkeypatch)
    sys.modules.pop("ei_vo.render.render_pyrender", None)
    pyrender_renderer = importlib.import_module("ei_vo.render.render_pyrender")
    monkeypatch.setattr(
        pyrender_renderer,
        "load_robot_model",
        lambda path, expected_dof=None: RobotModel(
            name="robot",
            joint_names=("joint1", "joint2"),
            limits=np.array([[-1.0, 1.0], [-1.0, 1.0]], dtype=float),
        ),
    )

    model_path = tmp_path / "robot.urdf"
    model_path.write_text("<robot/>", encoding="utf-8")

    with pytest.raises(ValueError, match="requires record_path"):
        pyrender_renderer.play(model_path, [[0.0, 0.0]])

    assert "loaded_path" not in captured


def test_pyrender_interactive_viewer_rejects_record_path(monkeypatch, tmp_path):
    _install_dummy_pyrender_runtime(monkeypatch)
    sys.modules.pop("ei_vo.render.render_pyrender", None)
    pyrender_renderer = importlib.import_module("ei_vo.render.render_pyrender")
    monkeypatch.setattr(
        pyrender_renderer,
        "load_robot_model",
        lambda path, expected_dof=None: RobotModel(
            name="robot",
            joint_names=("joint1", "joint2"),
            limits=np.array([[-1.0, 1.0], [-1.0, 1.0]], dtype=float),
        ),
    )

    model_path = tmp_path / "robot.urdf"
    model_path.write_text("<robot/>", encoding="utf-8")

    with pytest.raises(ValueError, match="interactive pyrender viewer"):
        pyrender_renderer.play(
            model_path,
            [[0.0, 0.0]],
            interactive=True,
            record_path=tmp_path / "clip.mp4",
        )


def test_pyrender_renderer_rejects_record_frames_dir_for_images(monkeypatch, tmp_path):
    _install_dummy_pyrender_runtime(monkeypatch)
    sys.modules.pop("ei_vo.render.render_pyrender", None)
    pyrender_renderer = importlib.import_module("ei_vo.render.render_pyrender")
    monkeypatch.setattr(
        pyrender_renderer,
        "load_robot_model",
        lambda path, expected_dof=None: RobotModel(
            name="robot",
            joint_names=("joint1", "joint2"),
            limits=np.array([[-1.0, 1.0], [-1.0, 1.0]], dtype=float),
        ),
    )

    model_path = tmp_path / "robot.urdf"
    model_path.write_text("<robot/>", encoding="utf-8")

    with pytest.raises(ValueError, match="record_frames_dir"):
        pyrender_renderer.play(
            model_path,
            [[0.0, 0.0]],
            record_path=tmp_path / "frame.png",
            record_frames_dir=tmp_path / "frames",
        )


def test_pyrender_renderer_saves_image(monkeypatch, tmp_path):
    captured = _install_dummy_pyrender_runtime(monkeypatch)
    sys.modules.pop("ei_vo.render.render_pyrender", None)
    pyrender_renderer = importlib.import_module("ei_vo.render.render_pyrender")
    monkeypatch.setattr(
        pyrender_renderer,
        "load_robot_model",
        lambda path, expected_dof=None: RobotModel(
            name="robot",
            joint_names=("joint1", "joint2"),
            limits=np.array([[-0.5, 0.5], [-0.25, 0.25]], dtype=float),
        ),
    )
    image_writes = []
    monkeypatch.setattr(
        pyrender_renderer,
        "_write_image",
        lambda path, frame: image_writes.append((pathlib.Path(path), np.asarray(frame).copy())),
    )

    model_path = tmp_path / "robot.urdf"
    model_path.write_text("<robot/>", encoding="utf-8")
    trajectory = Trajectory.from_positions([[1.0, -1.0], [0.0, 0.5]], dt=0.1)
    pyrender_renderer.play(
        model_path,
        trajectory,
        record_path=tmp_path / "frame.png",
        record_size=(320, 180),
        camera={"distance": 3.0, "azimuth": 90.0, "elevation": 10.0, "lookat": (0.0, 0.0, 0.2)},
    )

    assert captured["loaded_path"] == model_path.resolve().as_posix()
    assert len(captured["mesh_inputs"]) == 2
    assert all(entry["smooth"] is False for entry in captured["mesh_inputs"])
    assert image_writes[0][0] == tmp_path / "frame.png"
    assert image_writes[0][1].shape == (180, 320, 3)
    assert captured["renderer_deleted"] is True
    assert len(captured["renders"]) == 1
    assert len(captured["set_pose"]) == 2
    np.testing.assert_allclose(captured["set_pose"][0][:3, 3], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(captured["set_pose"][1][:3, 3], [0.0, 0.25, 0.0])


def test_pyrender_renderer_records_video(monkeypatch, tmp_path):
    captured = _install_dummy_pyrender_runtime(monkeypatch)
    sys.modules.pop("ei_vo.render.render_pyrender", None)
    pyrender_renderer = importlib.import_module("ei_vo.render.render_pyrender")
    monkeypatch.setattr(
        pyrender_renderer,
        "load_robot_model",
        lambda path, expected_dof=None: RobotModel(
            name="robot",
            joint_names=("joint1", "joint2"),
            limits=np.array([[-1.0, 1.0], [-1.0, 1.0]], dtype=float),
        ),
    )

    class DummyWriter:
        def __init__(self, path, *, fps, extension=".ppm", frames_dir=None, temp_prefix="ei_vo_frames_"):
            captured["writer"] = {
                "path": pathlib.Path(path),
                "fps": fps,
                "extension": extension,
                "frames_dir": None if frames_dir is None else pathlib.Path(frames_dir),
                "temp_prefix": temp_prefix,
                "frames": [],
                "closed": False,
            }

        def append_data(self, frame):
            captured["writer"]["frames"].append(np.asarray(frame).copy())

        def close(self):
            captured["writer"]["closed"] = True

    monkeypatch.setattr(pyrender_renderer, "FrameSequenceWriter", DummyWriter)

    model_path = tmp_path / "robot.urdf"
    model_path.write_text("<robot/>", encoding="utf-8")
    trajectory = Trajectory.from_positions([[0.0, 0.0], [0.1, 0.2], [0.2, 0.4]], dt=0.05)
    pyrender_renderer.play(
        model_path,
        trajectory,
        hz=20.0,
        record_path=tmp_path / "clip.mp4",
        record_frames_dir=tmp_path / "frames",
        record_size=(160, 90),
    )

    assert captured["writer"]["path"] == tmp_path / "clip.mp4"
    assert captured["writer"]["fps"] == 20.0
    assert captured["writer"]["extension"] == ".ppm"
    assert captured["writer"]["frames_dir"] == tmp_path / "frames"
    assert captured["writer"]["temp_prefix"] == "ei_vo_pyrender_"
    assert len(captured["writer"]["frames"]) == 3
    assert all(frame.shape == (90, 160, 3) for frame in captured["writer"]["frames"])
    assert captured["writer"]["closed"] is True
    assert captured["renderer_deleted"] is True
    assert len(captured["renders"]) == 3
    assert len(captured["set_pose"]) == 6


def test_pyrender_interactive_viewer_returns_final_camera(monkeypatch, tmp_path):
    captured = _install_dummy_pyrender_runtime(monkeypatch)
    sys.modules.pop("ei_vo.render.render_pyrender", None)
    pyrender_renderer = importlib.import_module("ei_vo.render.render_pyrender")
    monkeypatch.setattr(
        pyrender_renderer,
        "load_robot_model",
        lambda path, expected_dof=None: RobotModel(
            name="robot",
            joint_names=("joint1", "joint2"),
            limits=np.array([[-1.0, 1.0], [-1.0, 1.0]], dtype=float),
        ),
    )

    final_pose = np.eye(4, dtype=float)
    final_pose[:3, 3] = np.array([1.0, 2.0, 3.0], dtype=float)
    final_pose[:3, 2] = np.array([0.0, 0.0, 1.0], dtype=float)
    final_pose[:3, 0] = np.array([1.0, 0.0, 0.0], dtype=float)
    final_pose[:3, 1] = np.array([0.0, -1.0, 0.0], dtype=float)

    sleep_calls = {"count": 0}

    def fake_sleep(_seconds):
        sleep_calls["count"] += 1
        viewer = captured.get("viewer_instance")
        if viewer is not None and sleep_calls["count"] == 2:
            viewer._camera_node.matrix = final_pose.copy()
            viewer._trackball._n_target = np.array([1.0, 2.0, 1.0], dtype=float)
            viewer.is_active = False

    monkeypatch.setattr(pyrender_renderer.time, "sleep", fake_sleep)

    model_path = tmp_path / "robot.urdf"
    model_path.write_text("<robot/>", encoding="utf-8")
    trajectory = Trajectory.from_positions([[0.0, 0.0], [0.1, 0.2]], dt=0.1)
    camera = pyrender_renderer.play(
        model_path,
        trajectory,
        interactive=True,
        camera={"lookat": (0.0, 0.0, 0.0), "distance": 2.0},
    )

    assert len(captured["viewer_calls"]) == 1
    assert captured["viewer_calls"][0]["run_in_thread"] is True
    np.testing.assert_allclose(captured["viewer_calls"][0]["viewer_flags"]["view_center"], [0.0, 0.0, 0.0])
    assert captured["viewer_lock_acquires"] >= 1
    assert captured["viewer_lock_releases"] >= 1
    assert len(captured["set_pose"]) >= 2
    assert camera is not None
    assert pytest.approx(camera.distance) == 2.0
    assert pytest.approx(camera.azimuth) == 0.0
    assert pytest.approx(camera.elevation) == 90.0
    np.testing.assert_allclose(camera.lookat, [1.0, 2.0, 1.0])


def test_pyrender_interactive_viewer_surfaces_numpy_compat_errors(monkeypatch, tmp_path):
    _install_dummy_pyrender_runtime(monkeypatch)
    sys.modules.pop("ei_vo.render.render_pyrender", None)
    pyrender_renderer = importlib.import_module("ei_vo.render.render_pyrender")
    monkeypatch.setattr(
        pyrender_renderer,
        "load_robot_model",
        lambda path, expected_dof=None: RobotModel(
            name="robot",
            joint_names=("joint1", "joint2"),
            limits=np.array([[-1.0, 1.0], [-1.0, 1.0]], dtype=float),
        ),
    )

    class FailingViewer:
        def __init__(self, *args, **kwargs):
            del args, kwargs
            raise AttributeError("`np.infty` was removed in the NumPy 2.0 release. Use `np.inf` instead.")

    monkeypatch.setattr(sys.modules["pyrender"], "Viewer", FailingViewer)

    model_path = tmp_path / "robot.urdf"
    model_path.write_text("<robot/>", encoding="utf-8")

    with pytest.raises(RuntimeError, match="NumPy compatibility error"):
        pyrender_renderer.play(
            model_path,
            Trajectory.from_positions([[0.0, 0.0]], dt=0.1),
            interactive=True,
        )


def test_generic_play_dispatches_pyrender_renderer(monkeypatch, tmp_path):
    calls = {}
    sys.modules.pop("ei_vo.render.play", None)
    render_play = importlib.import_module("ei_vo.render.play")

    def fake_dispatch(renderer, /, **kwargs):
        calls["renderer"] = renderer
        calls["model_path"] = kwargs.pop("model_path")
        calls["traj"] = kwargs.pop("traj")
        calls["kwargs"] = kwargs

    monkeypatch.setattr(render_play, "dispatch_render", fake_dispatch)

    model_path = tmp_path / "robot.urdf"
    model_path.write_text("<robot/>", encoding="utf-8")
    render_play.play(
        model_path,
        [[0.0, 0.0]],
        renderer="pyrender",
        record_path=tmp_path / "clip.mp4",
    )

    assert calls["renderer"] == "pyrender"
    assert calls["model_path"] == model_path
    assert calls["kwargs"]["record_path"] == tmp_path / "clip.mp4"
