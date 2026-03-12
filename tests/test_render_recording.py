import importlib
import math
import pathlib
import sys
import types

import numpy as np
import pytest


def _import_render_module():
    sys.modules.pop("ei_vo.render.render_mj", None)
    sys.modules.pop("ei_vo.render", None)
    return importlib.import_module("ei_vo.render.render_mj")


def test_detect_arm_joints_skips_gripper_and_sorts(install_dummy_mujoco):
    install_dummy_mujoco(joint_names=["joint3", "gripper", "joint1", "finger", "joint2"])
    render_mj = _import_render_module()

    model = render_mj.mj.MjModel.from_xml_path("dummy.xml")
    arm = render_mj.detect_arm_joints(model)

    assert arm.joint_names == ("joint1", "joint2", "joint3")
    assert arm.qpos_addresses == (2, 4, 0)


def test_clamp_to_limits_uses_model_ranges(install_dummy_mujoco):
    install_dummy_mujoco(joint_names=["joint1", "joint2", "joint3"])
    render_mj = _import_render_module()

    model = render_mj.mj.MjModel.from_xml_path("dummy.xml")
    model.jnt_range = np.array([[-1.0, 1.0], [-0.5, 0.5], [-2.0, 2.0]])

    clipped = render_mj.clamp_to_limits(
        model,
        [0, 1, 2],
        np.array([[3.0, -1.0, 0.0], [-3.0, 1.0, 5.0]]),
    )

    np.testing.assert_allclose(clipped, [[1.0, -0.5, 0.0], [-1.0, 0.5, 2.0]])


def test_load_robot_model_returns_robot_metadata(install_dummy_mujoco):
    install_dummy_mujoco(joint_names=["joint1", "joint2", "joint3"])
    render_mj = _import_render_module()

    robot = render_mj.load_robot_model(pathlib.Path("dummy.xml"))

    assert robot.name == "dummy"
    assert robot.joint_names == ("joint1", "joint2", "joint3")
    assert robot.dof == 3


def test_load_robot_model_stages_nested_urdf_meshes(tmp_path, install_dummy_mujoco, monkeypatch):
    install_dummy_mujoco(joint_names=["joint1"])
    render_mj = _import_render_module()

    mesh_path = tmp_path / "meshes" / "collision" / "link0.stl"
    mesh_path.parent.mkdir(parents=True)
    mesh_path.write_text("solid link0\nendsolid link0\n", encoding="utf-8")

    urdf_path = tmp_path / "robot.urdf"
    urdf_path.write_text(
        """<?xml version="1.0"?>
<robot name="robot">
  <link name="base"/>
  <link name="link1">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="1"/>
      <inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/>
    </inertial>
    <collision>
      <geometry>
        <mesh filename="./meshes/collision/link0.stl"/>
      </geometry>
    </collision>
  </link>
  <joint name="joint1" type="revolute">
    <parent link="base"/>
    <child link="link1"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1" upper="1" effort="1" velocity="1"/>
  </joint>
</robot>
""",
        encoding="utf-8",
    )

    captured = {}
    original_from_xml_path = render_mj.mj.MjModel.from_xml_path

    def fake_from_xml_path(path):
        staged_path = pathlib.Path(path)
        captured["path"] = staged_path
        captured["urdf"] = staged_path.read_text(encoding="utf-8")
        captured["assets"] = {item.name for item in staged_path.parent.iterdir()}
        return original_from_xml_path(path)

    monkeypatch.setattr(render_mj.mj.MjModel, "from_xml_path", staticmethod(fake_from_xml_path))

    robot = render_mj.load_robot_model(urdf_path)

    assert robot.dof == 1
    assert captured["path"] != urdf_path
    assert captured["path"].name == urdf_path.name
    assert 'filename="asset_0000.stl"' in captured["urdf"]
    assert "asset_0000.stl" in captured["assets"]


def test_init_recording_defaults(tmp_path, install_dummy_mujoco, monkeypatch):
    install_dummy_mujoco()
    render_mj = _import_render_module()
    captured = {}

    class DummyWriter:
        def __init__(self, path, fps):
            captured["path"] = path
            captured["fps"] = fps

        def append_data(self, frame):
            pass

        def close(self):
            pass

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
    assert renderer.width == 1280
    assert renderer.height == 720
    assert isinstance(camera, render_mj.mj.MjvCamera)
    assert math.isclose(captured["fps"], 100.0)
    assert captured["path"].endswith("video.mp4")
    assert isinstance(writer, DummyWriter)


def test_init_recording_requires_video_backend(tmp_path, install_dummy_mujoco, monkeypatch):
    install_dummy_mujoco()
    render_mj = _import_render_module()

    def fail_get_writer(path, fps):
        raise ValueError("Could not find a backend to open the path.")

    imageio_v2 = types.ModuleType("imageio.v2")
    imageio_v2.get_writer = fail_get_writer
    imageio_module = types.ModuleType("imageio")
    imageio_module.v2 = imageio_v2

    monkeypatch.setitem(sys.modules, "imageio", imageio_module)
    monkeypatch.setitem(sys.modules, "imageio.v2", imageio_v2)

    with pytest.raises(RuntimeError, match=r"imageio\[ffmpeg\]"):
        render_mj._init_recording(
            render_mj.mj.MjModel.from_xml_path("dummy.xml"),
            dt=0.01,
            record_path=tmp_path / "video.mp4",
            record_fps=None,
            record_size=None,
        )


def test_play_trajectory_records_frames(tmp_path, install_dummy_mujoco, monkeypatch):
    install_dummy_mujoco()
    render_mj = _import_render_module()
    frames = []
    closed = {"renderer": False, "writer": False}
    captured_camera = {}

    class DummyRenderer:
        def __init__(self):
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

    def fake_init(model, dt, record_path, record_fps, record_size):
        camera = DummyCamera()
        captured_camera["camera"] = camera
        return DummyRenderer(), camera, DummyWriter()

    monkeypatch.setattr(render_mj, "_init_recording", fake_init)
    monkeypatch.setattr(render_mj.time, "sleep", lambda _: None)

    render_mj.play_trajectory(
        "model.xml",
        np.linspace(0.0, 1.0, 21, dtype=float).reshape(3, 7),
        playback=render_mj.PlaybackConfig(hz=10.0, slow=1.0, loop=False),
        recording=render_mj.RecordingConfig(path=tmp_path / "out.mp4"),
    )

    assert len(frames) == 3
    assert all(frame.dtype == np.uint8 for frame in frames)
    assert np.all(frames[0] == 127)
    assert captured_camera["camera"].distance == pytest.approx(3.75)
    assert closed["renderer"] and closed["writer"]
