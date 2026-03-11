import importlib
import math
import pathlib
import sys
import types

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def install_dummy_mujoco(monkeypatch):
    def _install(*, joint_names=None, viewer_states=(True, False)):
        names = list(joint_names or [f"joint{i + 1}" for i in range(7)])
        njnt = len(names)

        for module_name in list(sys.modules):
            if module_name == "mujoco" or module_name == "mujoco.viewer":
                sys.modules.pop(module_name, None)
            elif module_name.startswith("ei_vo.render"):
                sys.modules.pop(module_name, None)
            elif module_name == "examples.demo_mj":
                sys.modules.pop(module_name, None)

        dummy_mujoco = types.ModuleType("mujoco")

        class DummyMjModel:
            def __init__(self):
                self.njnt = njnt
                self.ngeom = 3
                self._joint_names = names
                self.jnt_type = np.array([0] * njnt)
                self.jnt_qposadr = np.arange(njnt)
                self.jnt_range = np.tile(np.array([[-math.pi, math.pi]]), (njnt, 1))
                self.geom_type = np.array([0, 1, 3])
                self.geom_size = np.array(
                    [
                        [0.05, 0.0, 0.0],
                        [0.10, 0.20, 0.30],
                        [0.04, 0.18, 0.0],
                    ]
                )
                self.geom_rgba = np.array(
                    [
                        [1.0, 0.1, 0.1, 1.0],
                        [0.1, 1.0, 0.1, 0.8],
                        [0.1, 0.1, 1.0, 1.0],
                    ]
                )
                self.stat = types.SimpleNamespace(center=np.array([0.1, -0.2, 0.3]), extent=2.5)
                self.vis = types.SimpleNamespace(
                    global_=types.SimpleNamespace(offwidth=640, offheight=480)
                )

            @staticmethod
            def from_xml_path(path):
                return DummyMjModel()

        class DummyMjData:
            def __init__(self, model):
                self.qpos = np.zeros(model.njnt)
                self.geom_xpos = np.zeros((model.ngeom, 3))
                self.geom_xmat = np.tile(np.eye(3).reshape(1, 9), (model.ngeom, 1))

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

        def mjv_default_camera(camera):
            camera.distance = 1.9
            camera.azimuth = 110.0
            camera.elevation = -20.0
            camera.lookat[:] = 0.0

        def mjv_default_free_camera(model, camera):
            camera.distance = model.stat.extent * 1.5
            camera.azimuth = 90.0
            camera.elevation = -45.0
            camera.lookat[:] = model.stat.center

        dummy_mujoco.MjModel = DummyMjModel
        dummy_mujoco.MjData = DummyMjData
        dummy_mujoco.Renderer = DummyRenderer
        dummy_mujoco.MjvCamera = DummyCamera
        dummy_mujoco.mjtGeom = types.SimpleNamespace(
            mjGEOM_SPHERE=0,
            mjGEOM_BOX=1,
            mjGEOM_CYLINDER=2,
            mjGEOM_CAPSULE=3,
            mjGEOM_ELLIPSOID=4,
        )
        dummy_mujoco.mjtJoint = types.SimpleNamespace(mjJNT_HINGE=0)
        dummy_mujoco.mjtObj = types.SimpleNamespace(mjOBJ_JOINT=0)
        dummy_mujoco.mj_id2name = lambda model, obj, joint_id: model._joint_names[joint_id]
        dummy_mujoco.mjv_defaultCamera = mjv_default_camera
        dummy_mujoco.mjv_defaultFreeCamera = mjv_default_free_camera

        def mj_forward(model, data):
            offset = float(data.qpos[0]) if data.qpos.size else 0.0
            for geom_id in range(model.ngeom):
                data.geom_xpos[geom_id] = np.array([0.2 * geom_id + offset, 0.05 * geom_id, 0.0])
                data.geom_xmat[geom_id] = np.eye(3).reshape(9)

        dummy_mujoco.mj_forward = mj_forward

        class DummyViewer:
            def __init__(self):
                self.cam = types.SimpleNamespace(
                    distance=0.0,
                    azimuth=0.0,
                    elevation=0.0,
                    lookat=np.zeros(3),
                )
                self._states = list(viewer_states)

            def is_running(self):
                return self._states.pop(0) if self._states else False

            def sync(self):
                pass

        class DummyViewerContext:
            def __enter__(self):
                return DummyViewer()

            def __exit__(self, exc_type, exc, tb):
                return False

        viewer_module = types.ModuleType("mujoco.viewer")
        viewer_module.launch_passive = lambda model, data: DummyViewerContext()

        monkeypatch.setitem(sys.modules, "mujoco", dummy_mujoco)
        monkeypatch.setitem(sys.modules, "mujoco.viewer", viewer_module)
        return types.SimpleNamespace(mujoco=dummy_mujoco, viewer=viewer_module)

    return _install


def import_fresh(module_name: str):
    sys.modules.pop(module_name, None)
    return importlib.import_module(module_name)
