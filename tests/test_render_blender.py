import importlib
import json
import pathlib
import sys

import numpy as np
import pytest

from ei_vo.core import Trajectory


def _import_blender_module():
    sys.modules.pop("ei_vo.render.render_blender", None)
    return importlib.import_module("ei_vo.render.render_blender")


def test_blender_renderer_invokes_headless_blender_for_video(monkeypatch, tmp_path):
    render_blender = _import_blender_module()
    captured = {}
    monkeypatch.setattr(render_blender.tempfile, "gettempdir", lambda: tmp_path.as_posix())

    class FakeProcess:
        def __init__(self, lines):
            self.stdout = iter(lines)
            self._return_code = 0

        def wait(self):
            return self._return_code

    def fake_popen(command, stdout, stderr, text):
        manifest_path = pathlib.Path(command[-1])
        captured["command"] = command
        captured["stdout"] = stdout
        captured["stderr"] = stderr
        captured["text"] = text
        captured["manifest"] = json.loads(manifest_path.read_text(encoding="utf-8"))
        frame_dir = pathlib.Path(captured["manifest"]["output"]["frame_dir"])
        frame_dir.mkdir(parents=True, exist_ok=True)
        (frame_dir / "0000000.png").write_bytes(b"frame0")
        (frame_dir / "0000001.png").write_bytes(b"frame1")
        return FakeProcess(["frame 0\n", "frame 1\n"])

    def fake_export(frame_dir, output_path, *, fps, extension, ffmpeg_path):
        captured["export"] = {
            "frame_dir": pathlib.Path(frame_dir),
            "output_path": pathlib.Path(output_path),
            "fps": fps,
            "extension": extension,
            "ffmpeg_path": ffmpeg_path,
        }
        return pathlib.Path(output_path)

    monkeypatch.setattr(render_blender, "find_blender_executable", lambda configured=None: "/usr/bin/blender")
    monkeypatch.setattr(render_blender, "find_ffmpeg_executable", lambda: "/usr/bin/ffmpeg")
    monkeypatch.setattr(render_blender.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(render_blender, "export_frame_sequence_to_video", fake_export)

    output_path = render_blender.play(
        "examples/models/three_dof_arm.urdf",
        Trajectory.from_positions([[0.0, 0.1, 0.2], [0.2, 0.3, 0.4]]),
        hz=60.0,
        slow=2.0,
        camera={"distance": 2.5, "azimuth": 90.0, "lookat": (0.0, 0.0, 0.4)},
        record_path=tmp_path / "render.mp4",
        record_size=(640, 480),
        engine="cycles",
        samples=12,
        floor=False,
    )

    assert pathlib.Path(output_path) == tmp_path / "render.mp4"
    assert captured["command"][:7] == [
        "/usr/bin/blender",
        "--background",
        "--factory-startup",
        "--python-exit-code",
        "1",
        "--python",
        pathlib.Path(render_blender.__file__).with_name("render_blender_script.py").as_posix(),
    ]
    assert captured["stdout"] is render_blender.subprocess.PIPE
    assert captured["stderr"] is render_blender.subprocess.STDOUT
    assert captured["text"] is True
    assert captured["manifest"]["model_path"].endswith("examples/models/three_dof_arm.urdf")
    assert captured["manifest"]["trajectory"] == [[0.0, 0.1, 0.2], [0.2, 0.3, 0.4]]
    assert captured["manifest"]["camera"]["distance"] == 2.5
    assert captured["manifest"]["camera"]["azimuth"] == 90.0
    assert captured["manifest"]["camera"]["lookat"] == [0.0, 0.0, 0.4]
    assert pathlib.Path(captured["manifest"]["scene_cache"]["path"]).parent == tmp_path / "ei_vo_blender_cache"
    assert captured["manifest"]["render"] == {
        "width": 640,
        "height": 480,
        "engine": "cycles",
        "samples": 12,
        "floor": False,
    }
    assert captured["manifest"]["output"]["kind"] == "video"
    assert captured["export"]["fps"] == pytest.approx(30.0)
    assert captured["export"]["extension"] == ".png"
    assert captured["export"]["ffmpeg_path"] == "/usr/bin/ffmpeg"


def test_blender_renderer_renders_still_image(monkeypatch, tmp_path):
    render_blender = _import_blender_module()
    captured = {}

    class FakeProcess:
        def __init__(self, lines):
            self.stdout = iter(lines)
            self._return_code = 0

        def wait(self):
            return self._return_code

    def fake_popen(command, stdout, stderr, text):
        manifest_path = pathlib.Path(command[-1])
        captured["manifest"] = json.loads(manifest_path.read_text(encoding="utf-8"))
        output_path = pathlib.Path(captured["manifest"]["output"]["path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"image")
        captured["stdout"] = stdout
        captured["stderr"] = stderr
        captured["text"] = text
        return FakeProcess(["saved image\n"])

    monkeypatch.setattr(render_blender, "find_blender_executable", lambda configured=None: "/usr/bin/blender")
    monkeypatch.setattr(render_blender.subprocess, "Popen", fake_popen)

    output_path = render_blender.play(
        "examples/models/three_dof_arm.urdf",
        [[0.0, 0.0, 0.0], [0.1, 0.2, 0.3]],
        record_path=tmp_path / "frame.png",
    )

    assert pathlib.Path(output_path) == tmp_path / "frame.png"
    assert (tmp_path / "frame.png").read_bytes() == b"image"
    assert captured["stdout"] is render_blender.subprocess.PIPE
    assert captured["stderr"] is render_blender.subprocess.STDOUT
    assert captured["text"] is True
    assert captured["manifest"]["output"] == {
        "kind": "image",
        "path": (tmp_path / "frame.png").as_posix(),
        "format": "PNG",
        "frame_index": 1,
    }


def test_blender_renderer_accepts_custom_image_frame_index(monkeypatch, tmp_path):
    render_blender = _import_blender_module()
    captured = {}

    class FakeProcess:
        def __init__(self, lines):
            self.stdout = iter(lines)
            self._return_code = 0

        def wait(self):
            return self._return_code

    def fake_popen(command, stdout, stderr, text):
        manifest_path = pathlib.Path(command[-1])
        captured["manifest"] = json.loads(manifest_path.read_text(encoding="utf-8"))
        output_path = pathlib.Path(captured["manifest"]["output"]["path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"image")
        return FakeProcess(["saved image\n"])

    monkeypatch.setattr(render_blender, "find_blender_executable", lambda configured=None: "/usr/bin/blender")
    monkeypatch.setattr(render_blender.subprocess, "Popen", fake_popen)

    render_blender.play(
        "examples/models/three_dof_arm.urdf",
        [[0.0, 0.0, 0.0], [0.1, 0.2, 0.3], [0.2, 0.4, 0.6]],
        record_path=tmp_path / "frame.png",
        image_frame_index=1,
    )

    assert captured["manifest"]["output"]["frame_index"] == 1


def test_blender_renderer_accepts_link_debug_output_path(monkeypatch, tmp_path):
    render_blender = _import_blender_module()
    captured = {}

    class FakeProcess:
        def __init__(self, lines):
            self.stdout = iter(lines)
            self._return_code = 0

        def wait(self):
            return self._return_code

    def fake_popen(command, stdout, stderr, text):
        manifest_path = pathlib.Path(command[-1])
        captured["manifest"] = json.loads(manifest_path.read_text(encoding="utf-8"))
        output_path = pathlib.Path(captured["manifest"]["output"]["path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"image")
        return FakeProcess(["saved image\n"])

    monkeypatch.setattr(render_blender, "find_blender_executable", lambda configured=None: "/usr/bin/blender")
    monkeypatch.setattr(render_blender.subprocess, "Popen", fake_popen)

    render_blender.play(
        "examples/models/three_dof_arm.urdf",
        [[0.0, 0.0, 0.0], [0.1, 0.2, 0.3]],
        record_path=tmp_path / "frame.png",
        debug_links_path=tmp_path / "debug" / "links.json",
    )

    assert captured["manifest"]["debug"] == {
        "links_path": (tmp_path / "debug" / "links.json").as_posix(),
    }


def test_blender_renderer_rejects_image_frame_index_for_video(tmp_path):
    render_blender = _import_blender_module()

    with pytest.raises(ValueError, match="image_frame_index"):
        render_blender.play(
            "examples/models/three_dof_arm.urdf",
            [[0.0, 0.0, 0.0], [0.1, 0.2, 0.3]],
            record_path=tmp_path / "render.mp4",
            image_frame_index=0,
        )


def test_blender_renderer_rejects_link_debug_output_for_video(tmp_path):
    render_blender = _import_blender_module()

    with pytest.raises(ValueError, match="debug_links_path"):
        render_blender.play(
            "examples/models/three_dof_arm.urdf",
            [[0.0, 0.0, 0.0], [0.1, 0.2, 0.3]],
            record_path=tmp_path / "render.mp4",
            debug_links_path=tmp_path / "links.json",
        )


def test_blender_renderer_reports_exit_code(monkeypatch, tmp_path):
    render_blender = _import_blender_module()

    class FakeProcess:
        def __init__(self):
            self.stdout = iter(["fatal error\n"])

        def wait(self):
            return -11

    monkeypatch.setattr(render_blender.subprocess, "Popen", lambda *args, **kwargs: FakeProcess())

    with pytest.raises(RuntimeError, match=r"exit code -11"):
        render_blender._run_blender_process("/usr/bin/blender", tmp_path / "scene.json")


def test_blender_renderer_falls_back_to_bpy_after_subprocess_failure(monkeypatch, tmp_path):
    render_blender = _import_blender_module()
    captured = {}

    monkeypatch.setattr(render_blender, "_find_blender_executable_optional", lambda configured=None: "/usr/bin/blender")
    monkeypatch.setattr(
        render_blender,
        "_run_blender_process",
        lambda blender_executable, manifest_path: (_ for _ in ()).throw(RuntimeError("crash")),
    )

    def fake_run_bpy_module(manifest, *, blender_executable):
        captured["manifest"] = manifest
        captured["blender_executable"] = blender_executable
        output_path = pathlib.Path(manifest["output"]["path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"image")

    monkeypatch.setattr(render_blender, "_run_bpy_module", fake_run_bpy_module)

    output_path = render_blender.play(
        "examples/models/three_dof_arm.urdf",
        [[0.0, 0.0, 0.0]],
        record_path=tmp_path / "frame.png",
    )

    assert pathlib.Path(output_path) == tmp_path / "frame.png"
    assert captured["blender_executable"] == "/usr/bin/blender"
    assert captured["manifest"]["output"]["kind"] == "image"


def test_blender_renderer_can_use_bpy_without_executable(monkeypatch, tmp_path):
    render_blender = _import_blender_module()
    captured = {}

    monkeypatch.setattr(render_blender, "_find_blender_executable_optional", lambda configured=None: None)

    def fake_run_bpy_module(manifest, *, blender_executable):
        captured["manifest"] = manifest
        captured["blender_executable"] = blender_executable
        output_path = pathlib.Path(manifest["output"]["path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"image")

    monkeypatch.setattr(render_blender, "_run_bpy_module", fake_run_bpy_module)

    output_path = render_blender.play(
        "examples/models/three_dof_arm.urdf",
        [[0.0, 0.0, 0.0]],
        record_path=tmp_path / "frame.png",
    )

    assert pathlib.Path(output_path) == tmp_path / "frame.png"
    assert captured["blender_executable"] is None
    assert captured["manifest"]["output"]["kind"] == "image"


def test_blender_renderer_accepts_custom_scene_cache_path(monkeypatch, tmp_path):
    render_blender = _import_blender_module()
    captured = {}

    class FakeProcess:
        def __init__(self, lines):
            self.stdout = iter(lines)
            self._return_code = 0

        def wait(self):
            return self._return_code

    def fake_popen(command, stdout, stderr, text):
        manifest_path = pathlib.Path(command[-1])
        captured["manifest"] = json.loads(manifest_path.read_text(encoding="utf-8"))
        output_path = pathlib.Path(captured["manifest"]["output"]["path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"image")
        return FakeProcess(["saved image\n"])

    monkeypatch.setattr(render_blender, "find_blender_executable", lambda configured=None: "/usr/bin/blender")
    monkeypatch.setattr(render_blender.subprocess, "Popen", fake_popen)

    render_blender.play(
        "examples/models/three_dof_arm.urdf",
        [[0.0, 0.0, 0.0]],
        record_path=tmp_path / "frame.png",
        scene_cache_path=tmp_path / "robot_cache",
    )

    assert captured["manifest"]["scene_cache"] == {
        "path": (tmp_path / "robot_cache.blend").as_posix(),
    }


def test_blender_renderer_can_disable_scene_cache(monkeypatch, tmp_path):
    render_blender = _import_blender_module()
    captured = {}

    class FakeProcess:
        def __init__(self, lines):
            self.stdout = iter(lines)
            self._return_code = 0

        def wait(self):
            return self._return_code

    def fake_popen(command, stdout, stderr, text):
        manifest_path = pathlib.Path(command[-1])
        captured["manifest"] = json.loads(manifest_path.read_text(encoding="utf-8"))
        output_path = pathlib.Path(captured["manifest"]["output"]["path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"image")
        return FakeProcess(["saved image\n"])

    monkeypatch.setattr(render_blender, "find_blender_executable", lambda configured=None: "/usr/bin/blender")
    monkeypatch.setattr(render_blender.subprocess, "Popen", fake_popen)

    render_blender.play(
        "examples/models/three_dof_arm.urdf",
        [[0.0, 0.0, 0.0]],
        record_path=tmp_path / "frame.png",
        scene_cache=False,
    )

    assert captured["manifest"]["scene_cache"] is None


def test_blender_renderer_resamples_video_trajectory_to_record_fps(monkeypatch, tmp_path):
    render_blender = _import_blender_module()
    captured = {}

    class FakeProcess:
        def __init__(self):
            self.stdout = iter(["rendered\n"])

        def wait(self):
            return 0

    def fake_popen(command, stdout, stderr, text):
        manifest_path = pathlib.Path(command[-1])
        captured["manifest"] = json.loads(manifest_path.read_text(encoding="utf-8"))
        frame_dir = pathlib.Path(captured["manifest"]["output"]["frame_dir"])
        frame_dir.mkdir(parents=True, exist_ok=True)
        for index in range(len(captured["manifest"]["trajectory"])):
            (frame_dir / f"{index:07d}.png").write_bytes(b"frame")
        return FakeProcess()

    monkeypatch.setattr(render_blender, "find_blender_executable", lambda configured=None: "/usr/bin/blender")
    monkeypatch.setattr(render_blender, "find_ffmpeg_executable", lambda: "/usr/bin/ffmpeg")
    monkeypatch.setattr(render_blender.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        render_blender,
        "export_frame_sequence_to_video",
        lambda *args, **kwargs: tmp_path / "render.mp4",
    )

    trajectory = Trajectory.from_positions(
        np.array([[0.0], [1.0], [2.0], [3.0], [4.0]], dtype=float),
        dt=0.25,
    )
    render_blender.play(
        "examples/models/three_dof_arm.urdf",
        Trajectory.from_positions(
            np.hstack([trajectory.q, trajectory.q, trajectory.q]),
            dt=0.25,
        ),
        hz=4.0,
        slow=1.0,
        record_fps=2.0,
        record_path=tmp_path / "render.mp4",
    )

    sampled = np.asarray(captured["manifest"]["trajectory"], dtype=float)
    assert sampled.shape[0] == 3
    np.testing.assert_allclose(sampled[:, 0], [0.0, 2.0, 4.0])


def test_blender_renderer_does_not_upsample_video_trajectory_for_high_record_fps(
    monkeypatch, tmp_path
):
    render_blender = _import_blender_module()
    captured = {}

    class FakeProcess:
        def __init__(self):
            self.stdout = iter(["rendered\n"])

        def wait(self):
            return 0

    def fake_popen(command, stdout, stderr, text):
        manifest_path = pathlib.Path(command[-1])
        captured["manifest"] = json.loads(manifest_path.read_text(encoding="utf-8"))
        frame_dir = pathlib.Path(captured["manifest"]["output"]["frame_dir"])
        frame_dir.mkdir(parents=True, exist_ok=True)
        for index in range(len(captured["manifest"]["trajectory"])):
            (frame_dir / f"{index:07d}.png").write_bytes(b"frame")
        return FakeProcess()

    monkeypatch.setattr(render_blender, "find_blender_executable", lambda configured=None: "/usr/bin/blender")
    monkeypatch.setattr(render_blender, "find_ffmpeg_executable", lambda: "/usr/bin/ffmpeg")
    monkeypatch.setattr(render_blender.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        render_blender,
        "export_frame_sequence_to_video",
        lambda *args, **kwargs: tmp_path / "render.mp4",
    )

    trajectory = np.array(
        [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]],
        dtype=float,
    )
    render_blender.play(
        "examples/models/three_dof_arm.urdf",
        Trajectory.from_positions(trajectory, dt=0.5),
        hz=2.0,
        slow=1.0,
        record_fps=60.0,
        record_path=tmp_path / "render.mp4",
    )

    sampled = np.asarray(captured["manifest"]["trajectory"], dtype=float)
    np.testing.assert_allclose(sampled, trajectory)


def test_blender_renderer_requires_record_path():
    render_blender = _import_blender_module()

    with pytest.raises(ValueError, match="record_path"):
        render_blender.play(
            "examples/models/three_dof_arm.urdf",
            [[0.0, 0.0, 0.0]],
        )


def test_find_blender_executable_supports_app_bundle(monkeypatch, tmp_path):
    render_blender = _import_blender_module()
    app_path = tmp_path / "Blender.app" / "Contents" / "MacOS"
    executable = app_path / "Blender"
    app_path.mkdir(parents=True)
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    executable.chmod(0o755)

    monkeypatch.setattr(render_blender.shutil, "which", lambda candidate: None)

    assert render_blender.find_blender_executable(tmp_path / "Blender.app") == executable.as_posix()
