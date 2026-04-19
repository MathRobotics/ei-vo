import importlib
import pathlib
import sys

import numpy as np
import pytest


def _import_render_module():
    sys.modules.pop("ei_vo.render.render_mj", None)
    sys.modules.pop("ei_vo.render", None)
    return importlib.import_module("ei_vo.render.render_mj")


def _import_recording_module():
    sys.modules.pop("ei_vo.core.recording", None)
    return importlib.import_module("ei_vo.core.recording")


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


def test_frame_sequence_writer_writes_ppm_and_invokes_ffmpeg(tmp_path, monkeypatch):
    recording = _import_recording_module()
    captured = {}

    def fake_run(command, cwd, capture_output, text):
        captured["command"] = command
        captured["cwd"] = pathlib.Path(cwd)
        captured["capture_output"] = capture_output
        captured["text"] = text
        pathlib.Path(command[-1]).write_bytes(b"video")
        return type("Result", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr(recording, "find_ffmpeg_executable", lambda: "/usr/bin/ffmpeg")
    monkeypatch.setattr(recording.subprocess, "run", fake_run)

    writer = recording.FrameSequenceWriter(tmp_path / "video", fps=100.0)
    frame_path = writer.append_data(np.full((2, 3, 3), 0.5, dtype=float))
    assert frame_path.name == "0000000.ppm"
    assert frame_path.read_bytes().startswith(b"P6\n3 2\n255\n")
    writer.close()
    assert captured["command"][:9] == [
        "/usr/bin/ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-start_number",
        "0",
        "-framerate",
        "100",
        "-i",
    ]
    assert captured["command"][9] == "%07d.ppm"
    assert captured["command"][-1].endswith("video.mp4")
    assert captured["cwd"].name.startswith("ei_vo_frames_")
    assert captured["capture_output"] is True
    assert captured["text"] is True


def test_frame_sequence_writer_requires_ffmpeg(tmp_path, monkeypatch):
    recording = _import_recording_module()
    monkeypatch.delenv("EI_VO_FFMPEG", raising=False)
    monkeypatch.setattr(recording.shutil, "which", lambda _: None)

    with pytest.raises(RuntimeError, match=r"ffmpeg"):
        recording.FrameSequenceWriter(tmp_path / "video.mp4", fps=30.0)


def test_export_frame_sequence_uses_absolute_output_for_relative_paths(tmp_path, monkeypatch):
    recording = _import_recording_module()
    captured = {}
    frame_dir = tmp_path / "frames"
    frame_dir.mkdir()

    def fake_run(command, cwd, capture_output, text):
        captured["command"] = command
        captured["cwd"] = pathlib.Path(cwd)
        return type("Result", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(recording.subprocess, "run", fake_run)

    recording.export_frame_sequence_to_video(
        frame_dir,
        pathlib.Path("tmp") / "out.mp4",
        fps=24.0,
        extension=".ppm",
        ffmpeg_path="/usr/bin/ffmpeg",
    )

    assert captured["cwd"] == frame_dir
    assert pathlib.Path(captured["command"][-1]).is_absolute()
    assert pathlib.Path(captured["command"][-1]) == (tmp_path / "tmp" / "out.mp4")


def test_frame_sequence_writer_persists_frames_under_requested_root(tmp_path, monkeypatch):
    recording = _import_recording_module()
    captured = {}

    def fake_run(command, cwd, capture_output, text):
        captured["cwd"] = pathlib.Path(cwd)
        return type("Result", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr(recording, "find_ffmpeg_executable", lambda: "/usr/bin/ffmpeg")
    monkeypatch.setattr(recording.subprocess, "run", fake_run)

    frames_root = tmp_path / "frames"
    writer = recording.FrameSequenceWriter(
        tmp_path / "video.mp4",
        fps=24.0,
        frames_dir=frames_root,
    )
    frame_path = writer.append_data(np.zeros((1, 2, 3), dtype=np.uint8))
    writer.close()

    assert frame_path.parent == frames_root / "video_frames"
    assert frame_path.exists()
    assert captured["cwd"] == frames_root / "video_frames"


def test_init_recording_defaults(tmp_path, install_dummy_mujoco, monkeypatch):
    install_dummy_mujoco()
    render_mj = _import_render_module()

    class DummyWriter:
        def __init__(self, path, *, fps, extension=".ppm", frames_dir=None, temp_prefix="ei_vo_frames_"):
            self.output_path = pathlib.Path(path)
            self.fps = fps
            self.extension = extension
            self.frames_dir = None if frames_dir is None else pathlib.Path(frames_dir)
            self.temp_prefix = temp_prefix

        def append_data(self, frame):
            pass

        def close(self):
            pass

    monkeypatch.setattr(render_mj, "FrameSequenceWriter", DummyWriter)

    renderer, camera, writer = render_mj._init_recording(
        render_mj.mj.MjModel.from_xml_path("dummy.xml"),
        dt=0.01,
        record_path=tmp_path / "video",
        record_fps=None,
        record_size=None,
        record_frames_dir=None,
    )

    assert isinstance(renderer, render_mj.mj.Renderer)
    assert renderer.width == 1280
    assert renderer.height == 720
    assert isinstance(camera, render_mj.mj.MjvCamera)
    assert writer.fps == pytest.approx(100.0)
    assert writer.output_path == tmp_path / "video.mp4"


def test_init_recording_requires_ffmpeg(tmp_path, install_dummy_mujoco, monkeypatch):
    install_dummy_mujoco()
    render_mj = _import_render_module()

    def fail_writer(*args, **kwargs):
        raise RuntimeError("Video export requires the 'ffmpeg' executable.")

    monkeypatch.setattr(render_mj, "FrameSequenceWriter", fail_writer)

    with pytest.raises(RuntimeError, match=r"ffmpeg"):
        render_mj._init_recording(
            render_mj.mj.MjModel.from_xml_path("dummy.xml"),
            dt=0.01,
            record_path=tmp_path / "video.mp4",
            record_fps=None,
            record_size=None,
            record_frames_dir=None,
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

    def fake_init(model, dt, record_path, record_fps, record_size, record_frames_dir):
        camera = DummyCamera()
        captured_camera["camera"] = camera
        captured_camera["frames_dir"] = record_frames_dir
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
    assert captured_camera["frames_dir"] is None
    assert closed["renderer"] and closed["writer"]
