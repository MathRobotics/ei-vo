import importlib
import importlib.util
import math
import pathlib
import sys
import types

import numpy as np
import pytest


@pytest.fixture
def render_mj(monkeypatch):
    root = pathlib.Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    for name in ["ei.render.render_mj", "ei.render", "ei"]:
        sys.modules.pop(name, None)

    dummy_mujoco = types.ModuleType("mujoco")

    class DummyMjModel:
        njnt = 7
        jnt_type = np.array([0] * 7)
        jnt_qposadr = np.arange(7)
        jnt_range = np.tile(np.array([[-np.pi, np.pi]]), (7, 1))

        @staticmethod
        def from_xml_path(path):
            return DummyMjModel()

    class DummyMjData:
        def __init__(self, model):
            self.qpos = np.zeros(7)

    class DummyRenderer:
        def __init__(self, model, height, width):
            self.model = model
            self.height = height
            self.width = width
            self.closed = False

        def update_scene(self, data, camera=None):
            pass

        def render(self):
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)

        def close(self):
            self.closed = True

    class DummyCamera:
        def __init__(self):
            self.distance = 0.0
            self.azimuth = 0.0
            self.elevation = 0.0
            self.lookat = np.zeros(3)

    def mjv_default_camera(cam):
        cam.distance = 1.9
        cam.azimuth = 110.0
        cam.elevation = -20.0
        cam.lookat[:] = 0.0

    dummy_mujoco.MjModel = DummyMjModel
    dummy_mujoco.MjData = DummyMjData
    dummy_mujoco.Renderer = DummyRenderer
    dummy_mujoco.MjvCamera = DummyCamera
    dummy_mujoco.mjtJoint = types.SimpleNamespace(mjJNT_HINGE=0)
    dummy_mujoco.mjtObj = types.SimpleNamespace(mjOBJ_JOINT=0)
    dummy_mujoco.mj_id2name = lambda m, obj, j_id: f"joint{j_id + 1}"
    dummy_mujoco.mjv_defaultCamera = mjv_default_camera
    dummy_mujoco.mj_forward = lambda m, d: None

    class DummyViewer:
        def __init__(self):
            self.cam = types.SimpleNamespace(
                distance=1.9,
                azimuth=110.0,
                elevation=-20.0,
                lookat=np.zeros(3),
            )
            self._states = [True, False]

        def is_running(self):
            return self._states.pop(0) if self._states else False

        def sync(self):
            pass

    class DummyViewerContext:
        def __enter__(self):
            return DummyViewer()

        def __exit__(self, exc_type, exc, tb):
            pass

    viewer_module = types.ModuleType("mujoco.viewer")
    viewer_module.launch_passive = lambda model, data: DummyViewerContext()

    monkeypatch.setitem(sys.modules, "mujoco", dummy_mujoco)
    monkeypatch.setitem(sys.modules, "mujoco.viewer", viewer_module)

    package_root = root / "ei-vo"

    ei_pkg = types.ModuleType("ei")
    ei_pkg.__path__ = [str(package_root)]
    core_pkg = types.ModuleType("ei.core")
    core_pkg.__path__ = [str(package_root / "core")]
    render_pkg = types.ModuleType("ei.render")
    render_pkg.__path__ = [str(package_root / "render")]

    monkeypatch.setitem(sys.modules, "ei", ei_pkg)
    monkeypatch.setitem(sys.modules, "ei.core", core_pkg)
    monkeypatch.setitem(sys.modules, "ei.render", render_pkg)

    core_spec = importlib.util.spec_from_file_location(
        "ei.core.core", package_root / "core" / "core.py"
    )
    core_module = importlib.util.module_from_spec(core_spec)
    monkeypatch.setitem(sys.modules, "ei.core.core", core_module)
    core_spec.loader.exec_module(core_module)

    render_spec = importlib.util.spec_from_file_location(
        "ei.render.render_mj", package_root / "render" / "render_mj.py"
    )
    module = importlib.util.module_from_spec(render_spec)
    monkeypatch.setitem(sys.modules, "ei.render.render_mj", module)
    render_spec.loader.exec_module(module)

    ei_pkg.render = render_pkg
    render_pkg.render_mj = module
    return module


def test_init_recording_defaults(tmp_path, render_mj, monkeypatch):
    captured = {}

    class DummyWriter:
        def __init__(self, path, fps):
            captured["path"] = path
            captured["fps"] = fps
            self.closed = False

        def append_data(self, frame):
            pass

        def close(self):
            self.closed = True

    imageio_v2 = types.ModuleType("imageio.v2")
    imageio_v2.get_writer = lambda path, fps: DummyWriter(path, fps)
    imageio_module = types.ModuleType("imageio")
    imageio_module.v2 = imageio_v2

    monkeypatch.setitem(sys.modules, "imageio", imageio_module)
    monkeypatch.setitem(sys.modules, "imageio.v2", imageio_v2)

    renderer, camera, writer = render_mj._init_recording(
        render_mj.mj.MjModel.from_xml_path("dummy.xml"),
        dt=0.01,
        record_path=tmp_path / "video",
        record_fps=None,
        record_size=None,
    )

    assert isinstance(renderer, render_mj.mj.Renderer)
    assert renderer.width == 1280 and renderer.height == 720
    assert isinstance(camera, render_mj.mj.MjvCamera)
    assert math.isclose(captured["fps"], 100.0)
    assert captured["path"].endswith("video.mp4")
    assert isinstance(writer, DummyWriter)


def test_play_records_frames(tmp_path, render_mj, monkeypatch):
    frames = []
    closed = {"renderer": False, "writer": False}

    class DummyRenderer:
        def __init__(self, *args, **kwargs):
            self.height = 3
            self.width = 4

        def update_scene(self, data, camera=None):
            pass

        def render(self):
            return np.full((self.height, self.width, 3), 0.5, dtype=float)

        def close(self):
            closed["renderer"] = True

    class DummyCamera:
        def __init__(self):
            self.distance = 0.0
            self.azimuth = 0.0
            self.elevation = 0.0
            self.lookat = np.zeros(3)

    class DummyWriter:
        def append_data(self, frame):
            frames.append(frame)

        def close(self):
            closed["writer"] = True

    captured_camera = {}

    def fake_init(model, dt, record_path, record_fps, record_size):
        camera = DummyCamera()
        captured_camera["camera"] = camera
        return DummyRenderer(), camera, DummyWriter()

    monkeypatch.setattr(render_mj, "_init_recording", fake_init)
    monkeypatch.setattr(render_mj.time, "sleep", lambda _: None)

    traj = types.SimpleNamespace(q=np.linspace(0.0, 1.0, 21, dtype=float).reshape(3, 7))

    render_mj.play(
        "model.xml",
        traj,
        slow=1.0,
        hz=10.0,
        loop=False,
        record_path=tmp_path / "out.mp4",
        record_fps=None,
        record_size=None,
    )

    assert len(frames) == traj.q.shape[0]
    assert all(frame.dtype == np.uint8 for frame in frames)
    assert np.all(frames[0] == 127)  # 0.5 * 255 rounded down
    assert captured_camera["camera"].distance == pytest.approx(1.9)
    assert closed["renderer"] and closed["writer"]
