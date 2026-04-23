import importlib
import pathlib
import sys
import types

import numpy as np
import pytest

from ei_vo import RenderSpec
from ei_vo.core import Trajectory


def _write_demo_urdf(path: pathlib.Path) -> pathlib.Path:
    path.write_text(
        """<?xml version="1.0"?>
<robot name="demo_arm">
  <link name="base">
    <visual>
      <origin xyz="0 0 0.05" rpy="0 0 0"/>
      <geometry><box size="0.2 0.2 0.1"/></geometry>
    </visual>
  </link>
  <link name="link1">
    <visual>
      <origin xyz="0 0 0.15" rpy="0 0 0"/>
      <geometry><cylinder radius="0.03" length="0.3"/></geometry>
    </visual>
  </link>
  <link name="ee">
    <visual>
      <geometry><sphere radius="0.02"/></geometry>
    </visual>
  </link>
  <joint name="joint1" type="revolute">
    <parent link="base"/>
    <child link="link1"/>
    <origin xyz="0 0 0.1" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.0" upper="1.0" effort="1" velocity="1"/>
  </joint>
  <joint name="joint2" type="revolute">
    <parent link="link1"/>
    <child link="ee"/>
    <origin xyz="0 0 0.3" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-0.5" upper="0.5" effort="1" velocity="1"/>
  </joint>
</robot>
""",
        encoding="utf-8",
    )
    return path


def _install_dummy_matplotlib(monkeypatch, *, backend="tkagg"):
    captured = {}

    class DummyCanvas:
        def draw(self):
            captured["draw_calls"] = captured.get("draw_calls", 0) + 1

        def get_width_height(self):
            return (4, 3)

        def buffer_rgba(self):
            return np.full((3, 4, 4), 127, dtype=np.uint8)

    class DummyAxis:
        def clear(self):
            captured["clear_calls"] = captured.get("clear_calls", 0) + 1

        def plot(self, x, y, z=None, **kwargs):
            captured.setdefault("plots", []).append(
                (
                    np.asarray(x),
                    np.asarray(y),
                    None if z is None else np.asarray(z),
                    dict(kwargs),
                )
            )

        def scatter(self, x, y, z, **kwargs):
            captured.setdefault("scatters", []).append(
                (
                    np.asarray(x),
                    np.asarray(y),
                    np.asarray(z),
                    dict(kwargs),
                )
            )

        def set_xlabel(self, value):
            captured["xlabel"] = value

        def set_ylabel(self, value):
            captured["ylabel"] = value

        def set_zlabel(self, value):
            captured["zlabel"] = value

        def set_title(self, value):
            captured["title"] = value

        def grid(self, *args, **kwargs):
            captured["grid"] = True

        def set_xlim(self, left, right):
            captured["xlim"] = (left, right)

        def set_ylim(self, bottom, top):
            captured["ylim"] = (bottom, top)

        def set_zlim(self, bottom, top):
            captured["zlim"] = (bottom, top)

        def set_box_aspect(self, value):
            captured["box_aspect"] = tuple(value)

        def view_init(self, elev=None, azim=None):
            captured["view_init"] = (elev, azim)

    class DummyFigure:
        def __init__(self):
            self.canvas = DummyCanvas()

        def add_subplot(self, *args, projection=None):
            captured["projection"] = projection
            return DummyAxis()

        def savefig(self, path, dpi):
            captured["savefig"] = (path, dpi)

    def figure(figsize=None, constrained_layout=None):
        captured["figsize"] = figsize
        captured["constrained_layout"] = constrained_layout
        return DummyFigure()

    def show():
        captured["shown"] = captured.get("shown", 0) + 1

    def pause(interval):
        captured.setdefault("pauses", []).append(interval)

    def close(figure):
        captured["closed"] = True

    pyplot = types.ModuleType("matplotlib.pyplot")
    pyplot.figure = figure
    pyplot.show = show
    pyplot.pause = pause
    pyplot.close = close

    matplotlib = types.ModuleType("matplotlib")
    matplotlib.pyplot = pyplot
    matplotlib.get_backend = lambda: backend

    monkeypatch.setitem(sys.modules, "matplotlib", matplotlib)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", pyplot)
    return captured


def _install_dummy_meshcat(monkeypatch):
    captured = {
        "objects": {},
        "properties": {},
        "transforms": {},
        "animations": [],
        "opened": 0,
        "html": "<html>meshcat-scene</html>",
    }

    class DummyAnimationFrame:
        def __init__(self, animation, path, frame):
            self.animation = animation
            self.path = path
            self.frame = frame

        def set_transform(self, transform):
            self.animation.frames.setdefault(self.path, []).append((self.frame, np.asarray(transform)))

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class DummyAnimation:
        def __init__(self, default_framerate=30):
            self.default_framerate = default_framerate
            self.frames = {}

        def at_frame(self, visualizer, frame):
            return DummyAnimationFrame(self, visualizer.path, frame)

    class DummyNode:
        def __init__(self, path=""):
            self.path = path

        def __getitem__(self, key):
            if key.startswith("/"):
                child_path = key
            else:
                child_path = f"{self.path}/{key}" if self.path else key
            return DummyNode(child_path)

        def set_object(self, obj, material=None):
            captured["objects"][self.path] = (obj, material)

        def set_transform(self, transform):
            captured["transforms"].setdefault(self.path, []).append(np.asarray(transform))

        def set_property(self, key, value):
            captured["properties"][(self.path, key)] = value

        def open(self):
            captured["opened"] += 1
            return self

        def set_animation(self, animation, play=True, repetitions=1):
            captured["animations"].append(
                {
                    "animation": animation,
                    "play": play,
                    "repetitions": repetitions,
                    "path": self.path,
                }
            )

        def static_html(self):
            return captured["html"]

    class DummyVisualizer(DummyNode):
        def __init__(self, zmq_url=None, **kwargs):
            del kwargs
            super().__init__()
            captured["zmq_url"] = zmq_url

    animation = types.ModuleType("meshcat.animation")
    animation.Animation = DummyAnimation

    meshcat = types.ModuleType("meshcat")
    meshcat.Visualizer = DummyVisualizer
    meshcat.animation = animation

    monkeypatch.setitem(sys.modules, "meshcat", meshcat)
    monkeypatch.setitem(sys.modules, "meshcat.animation", animation)
    return captured


def _install_dummy_pinocchio_meshcat(monkeypatch):
    captured = {"build_calls": [], "loaded_roots": [], "displayed": []}

    class DummyModel:
        nq = 2

        def __init__(self):
            self.lowerPositionLimit = np.array([-0.5, -0.25], dtype=float)
            self.upperPositionLimit = np.array([0.5, 0.25], dtype=float)

    class DummyGeometryModel:
        def __init__(self):
            self.geometryObjects = [types.SimpleNamespace(name="base"), types.SimpleNamespace(name="ee")]

    class DummyMeshcatVisualizer:
        def __init__(self, model, collision_model=None, visual_model=None, **kwargs):
            del kwargs
            self.model = model
            self.collision_model = collision_model
            self.visual_model = visual_model
            self.viewer = None
            self.viewerRootNodeName = None
            self.viewerVisualGroupName = None
            self.viewerCollisionGroupName = None

        def initViewer(self, viewer=None, open=False, loadModel=False, zmq_url=None):
            del loadModel, zmq_url
            self.viewer = viewer
            if open:
                self.viewer.open()

        def loadViewerModel(
            self,
            rootNodeName="pinocchio",
            color=None,
            collision_color=None,
            visual_color=None,
        ):
            del color, collision_color, visual_color
            self.viewerRootNodeName = rootNodeName
            self.viewerVisualGroupName = f"{rootNodeName}/visuals"
            self.viewerCollisionGroupName = f"{rootNodeName}/collisions"
            captured["loaded_roots"].append(rootNodeName)
            self.viewer[f"{self.viewerVisualGroupName}/base"].set_object("base")
            self.viewer[f"{self.viewerVisualGroupName}/ee"].set_object("ee")
            self.viewer[self.viewerCollisionGroupName].set_property("visible", False)
            self.viewer[self.viewerVisualGroupName].set_property("visible", True)

        def display(self, q=None):
            row = np.asarray(q, dtype=float)
            captured["displayed"].append(row.copy())
            self.viewer[f"{self.viewerVisualGroupName}/base"].set_transform(np.eye(4))
            transform = np.eye(4)
            transform[0, 3] = row.sum()
            self.viewer[f"{self.viewerVisualGroupName}/ee"].set_transform(transform)

    def build_models_from_urdf(path, **kwargs):
        captured["build_calls"].append({"path": path, "kwargs": kwargs})
        return DummyModel(), DummyGeometryModel(), DummyGeometryModel()

    pinocchio = types.ModuleType("pinocchio")
    pinocchio.buildModelsFromUrdf = build_models_from_urdf

    visualize = types.ModuleType("pinocchio.visualize")
    visualize.MeshcatVisualizer = DummyMeshcatVisualizer
    pinocchio.visualize = visualize

    monkeypatch.setitem(sys.modules, "pinocchio", pinocchio)
    monkeypatch.setitem(sys.modules, "pinocchio.visualize", visualize)
    return captured


def test_render_package_is_lazy_without_optional_renderers():
    sys.modules.pop("ei_vo.render", None)
    sys.modules.pop("meshcat", None)
    sys.modules.pop("pinocchio", None)
    sys.modules.pop("pyrender", None)

    render = importlib.import_module("ei_vo.render")

    assert render.available_renderers() == ("matplotlib", "meshcat", "pyrender")


def test_matplotlib_renderer_saves_image(monkeypatch, tmp_path):
    captured = _install_dummy_matplotlib(monkeypatch)
    sys.modules.pop("ei_vo.render.render_matplotlib", None)
    matplotlib_renderer = importlib.import_module("ei_vo.render.render_matplotlib")

    trajectory = Trajectory.from_positions(np.array([[0.0, 0.0], [0.4, 0.2], [0.8, -0.6]], dtype=float), dt=0.1)
    matplotlib_renderer.play(
        _write_demo_urdf(tmp_path / "robot.urdf"),
        trajectory,
        hz=10.0,
        show=False,
        record_path=tmp_path / "trajectory_plot",
    )

    assert captured["projection"] == "3d"
    assert len(captured["plots"]) >= 2
    assert len(captured["scatters"]) >= 1
    assert captured["savefig"][0].endswith("trajectory_plot.png")
    assert captured["xlabel"] == "x [m]"
    assert captured["ylabel"] == "y [m]"
    assert captured["zlabel"] == "z [m]"
    assert captured["title"].endswith("(3/3)")
    assert captured["box_aspect"] == (1.0, 1.0, 1.0)
    assert captured["closed"] is True


def test_matplotlib_renderer_animates_geometry_when_shown(monkeypatch, tmp_path):
    captured = _install_dummy_matplotlib(monkeypatch)
    sys.modules.pop("ei_vo.render.render_matplotlib", None)
    matplotlib_renderer = importlib.import_module("ei_vo.render.render_matplotlib")

    matplotlib_renderer.play(
        _write_demo_urdf(tmp_path / "robot.urdf"),
        [[0.0, 0.2], [0.4, 0.1], [0.8, -0.2]],
        hz=4.0,
        show=True,
        title="Animated Geometry",
    )

    assert captured["shown"] == 1
    assert len(captured["pauses"]) == 2
    assert captured["clear_calls"] == 3
    assert len(captured["plots"]) >= 6
    assert captured["title"] == "Animated Geometry (3/3)"


def test_matplotlib_renderer_skips_live_show_for_non_interactive_backend(monkeypatch, tmp_path):
    captured = _install_dummy_matplotlib(monkeypatch, backend="agg")
    sys.modules.pop("ei_vo.render.render_matplotlib", None)
    matplotlib_renderer = importlib.import_module("ei_vo.render.render_matplotlib")

    matplotlib_renderer.play(
        _write_demo_urdf(tmp_path / "robot.urdf"),
        [[0.0, 0.2], [0.4, 0.1], [0.8, -0.2]],
        hz=4.0,
        show=True,
        title="Headless Geometry",
    )

    assert "shown" not in captured
    assert "pauses" not in captured
    assert captured["clear_calls"] == 1
    assert captured["title"] == "Headless Geometry (3/3)"


def test_matplotlib_renderer_records_mp4(monkeypatch, tmp_path):
    captured = _install_dummy_matplotlib(monkeypatch)
    sys.modules.pop("ei_vo.render.render_matplotlib", None)
    matplotlib_renderer = importlib.import_module("ei_vo.render.render_matplotlib")

    class DummyWriter:
        def __init__(self, path, *, fps, extension=".ppm", frames_dir=None, temp_prefix="ei_vo_frames_"):
            captured["video_path"] = pathlib.Path(path).as_posix()
            captured["video_fps"] = fps
            captured["video_frames"] = []
            captured["video_closed"] = False
            captured["video_extension"] = extension
            captured["video_frames_dir"] = None if frames_dir is None else pathlib.Path(frames_dir)
            captured["video_temp_prefix"] = temp_prefix

        def append_data(self, frame):
            captured["video_frames"].append(np.asarray(frame))

        def close(self):
            captured["video_closed"] = True

    monkeypatch.setattr(matplotlib_renderer, "FrameSequenceWriter", DummyWriter)

    matplotlib_renderer.play(
        _write_demo_urdf(tmp_path / "robot.urdf"),
        [[0.0, 0.2], [0.4, 0.1], [0.8, -0.2]],
        hz=6.0,
        show=False,
        record_path=tmp_path / "trajectory.mp4",
    )

    assert captured["video_path"].endswith("trajectory.mp4")
    assert captured["video_fps"] == 6.0
    assert captured["video_extension"] == ".ppm"
    assert captured["video_frames_dir"] is None
    assert captured["video_temp_prefix"] == "ei_vo_matplotlib_"
    assert len(captured["video_frames"]) == 3
    assert all(frame.shape == (3, 4, 3) for frame in captured["video_frames"])
    assert captured["video_closed"] is True


def test_generic_play_dispatches_matplotlib_renderer(monkeypatch, tmp_path):
    captured = _install_dummy_matplotlib(monkeypatch)
    sys.modules.pop("ei_vo.render.render_matplotlib", None)
    sys.modules.pop("ei_vo.render", None)
    render = importlib.import_module("ei_vo.render")

    render.play(
        _write_demo_urdf(tmp_path / "robot.urdf"),
        [[0.0, 0.1], [0.2, 0.3]],
        hz=2.0,
        renderer="matplotlib",
        show=False,
        title="Angles",
    )

    assert captured["title"] == "Angles (2/2)"


def test_generic_play_accepts_render_spec(monkeypatch, tmp_path):
    captured = _install_dummy_matplotlib(monkeypatch)
    sys.modules.pop("ei_vo.render.render_matplotlib", None)
    sys.modules.pop("ei_vo.render", None)
    render = importlib.import_module("ei_vo.render")

    render.play(
        _write_demo_urdf(tmp_path / "robot.urdf"),
        [[0.0, 0.1], [0.2, 0.3]],
        hz=2.0,
        renderer=RenderSpec("plot", options={"show": False, "title": "Configured Plot"}),
    )

    assert captured["title"] == "Configured Plot (2/2)"


def test_matplotlib_renderer_requires_model(monkeypatch):
    _install_dummy_matplotlib(monkeypatch)
    sys.modules.pop("ei_vo.render.render_matplotlib", None)
    matplotlib_renderer = importlib.import_module("ei_vo.render.render_matplotlib")

    with pytest.raises(ValueError, match="--model is required"):
        matplotlib_renderer.play(None, [[0.0, 0.0], [1.0, 1.0]], hz=2.0)


def test_meshcat_record_path_resolves_to_html():
    sys.modules.pop("ei_vo.render.render_meshcat", None)
    meshcat_renderer = importlib.import_module("ei_vo.render.render_meshcat")

    html_path = meshcat_renderer._resolve_record_html_path(pathlib.Path("scene.html"))
    auto_html_path = meshcat_renderer._resolve_record_html_path(pathlib.Path("scene"))
    converted_html_path = meshcat_renderer._resolve_record_html_path(pathlib.Path("scene.mp4"))

    assert html_path == pathlib.Path("scene.html")
    assert auto_html_path == pathlib.Path("scene.html")
    assert converted_html_path == pathlib.Path("scene.html")


def test_meshcat_visualizer_uses_default_server(monkeypatch):
    captured = _install_dummy_meshcat(monkeypatch)
    sys.modules.pop("ei_vo.render.render_meshcat", None)
    meshcat_renderer = importlib.import_module("ei_vo.render.render_meshcat")

    visualizer = meshcat_renderer._create_visualizer(importlib.import_module("meshcat"))

    assert visualizer.path == ""
    assert captured["zmq_url"] is None


def test_meshcat_renderer_saves_standalone_html(monkeypatch, tmp_path):
    captured = _install_dummy_meshcat(monkeypatch)
    pinocchio_captured = _install_dummy_pinocchio_meshcat(monkeypatch)
    sys.modules.pop("ei_vo.render.render_meshcat", None)
    meshcat_renderer = importlib.import_module("ei_vo.render.render_meshcat")
    monkeypatch.setattr(meshcat_renderer.time, "sleep", lambda _: None)
    opened_html = []
    monkeypatch.setattr(meshcat_renderer, "_open_standalone_recording_html", lambda path: opened_html.append(path))

    model_path = _write_demo_urdf(tmp_path / "robot.urdf")
    trajectory = Trajectory.from_positions(np.array([[1.0, -1.0], [-1.0, 1.0]], dtype=float), dt=0.1)
    output_path = tmp_path / "scene"
    meshcat_renderer.play(
        model_path,
        trajectory,
        hz=20.0,
        open_browser=True,
        record_path=output_path,
    )

    expected_package_dirs = [
        candidate.as_posix()
        for candidate in (model_path.parent.resolve(), *model_path.parent.resolve().parents)
    ]
    assert pinocchio_captured["build_calls"] == [
        {
            "path": model_path.resolve().as_posix(),
            "kwargs": {"package_dirs": expected_package_dirs},
        }
    ]
    assert pinocchio_captured["loaded_roots"] == ["pinocchio"]
    np.testing.assert_allclose(pinocchio_captured["displayed"][0], [0.5, -0.25])
    np.testing.assert_allclose(pinocchio_captured["displayed"][1], [-0.5, 0.25])
    assert "pinocchio/visuals/base" in captured["objects"]
    assert "pinocchio/visuals/ee" in captured["objects"]
    assert len(captured["transforms"]["pinocchio/visuals/ee"]) == trajectory.steps
    assert len(captured["animations"]) == 1
    assert len(captured["animations"][0]["animation"].frames["pinocchio/visuals/ee"]) == trajectory.steps
    assert output_path.with_suffix(".html").read_text(encoding="utf-8") == captured["html"]
    assert opened_html == [output_path.with_suffix(".html")]


def test_meshcat_renderer_applies_lookat_to_urdf_visualizer(monkeypatch, tmp_path):
    captured = _install_dummy_meshcat(monkeypatch)
    _install_dummy_pinocchio_meshcat(monkeypatch)
    sys.modules.pop("ei_vo.render.render_meshcat", None)
    meshcat_renderer = importlib.import_module("ei_vo.render.render_meshcat")
    monkeypatch.setattr(meshcat_renderer.time, "sleep", lambda _: None)

    model_path = _write_demo_urdf(tmp_path / "robot.urdf")
    meshcat_renderer.play(
        model_path,
        Trajectory.from_positions(np.zeros((1, 2), dtype=float), dt=0.1),
        hz=20.0,
        open_browser=False,
        camera={"lookat": (0.5, -0.25, 1.2)},
    )

    expected_transform = np.eye(4)
    expected_transform[:3, 3] = np.array([0.5, -0.25, 1.2], dtype=float)
    np.testing.assert_allclose(captured["transforms"]["/Cameras/default"][-1], expected_transform)
    np.testing.assert_allclose(
        captured["properties"][("/Cameras/default/rotated/<object>", "position")],
        [3.0, 1.0, 0.0],
    )


def test_meshcat_renderer_rejects_record_size(monkeypatch, tmp_path):
    _install_dummy_meshcat(monkeypatch)
    _install_dummy_pinocchio_meshcat(monkeypatch)
    sys.modules.pop("ei_vo.render.render_meshcat", None)
    meshcat_renderer = importlib.import_module("ei_vo.render.render_meshcat")

    with pytest.raises(ValueError, match="record_size"):
        meshcat_renderer.play(
            _write_demo_urdf(tmp_path / "robot.urdf"),
            Trajectory.from_positions(np.zeros((1, 2), dtype=float), dt=0.1),
            record_path="scene.html",
            record_size=(640, 360),
        )


def test_meshcat_renderer_rejects_record_frames_dir(monkeypatch, tmp_path):
    _install_dummy_meshcat(monkeypatch)
    _install_dummy_pinocchio_meshcat(monkeypatch)
    sys.modules.pop("ei_vo.render.render_meshcat", None)
    meshcat_renderer = importlib.import_module("ei_vo.render.render_meshcat")

    with pytest.raises(ValueError, match="record_frames_dir"):
        meshcat_renderer.play(
            _write_demo_urdf(tmp_path / "robot.urdf"),
            Trajectory.from_positions(np.zeros((1, 2), dtype=float), dt=0.1),
            record_path="scene.html",
            record_frames_dir="frames",
        )


def test_meshcat_renderer_rejects_non_urdf(monkeypatch, tmp_path):
    _install_dummy_meshcat(monkeypatch)
    _install_dummy_pinocchio_meshcat(monkeypatch)
    sys.modules.pop("ei_vo.render.render_meshcat", None)
    meshcat_renderer = importlib.import_module("ei_vo.render.render_meshcat")
    model_path = tmp_path / "robot.xml"
    model_path.write_text("<robot/>", encoding="utf-8")

    with pytest.raises(ValueError, match="only supports URDF"):
        meshcat_renderer.play(model_path, Trajectory.from_positions(np.zeros((1, 2), dtype=float), dt=0.1))


def test_generic_play_dispatches_meshcat_renderer(monkeypatch, tmp_path):
    captured = _install_dummy_meshcat(monkeypatch)
    _install_dummy_pinocchio_meshcat(monkeypatch)
    sys.modules.pop("ei_vo.render.render_meshcat", None)
    sys.modules.pop("ei_vo.render", None)
    render = importlib.import_module("ei_vo.render")
    meshcat_renderer = importlib.import_module("ei_vo.render.render_meshcat")
    monkeypatch.setattr(meshcat_renderer.time, "sleep", lambda _: None)

    render.play(
        _write_demo_urdf(tmp_path / "robot.urdf"),
        np.array([[0.0, 0.0], [0.2, 0.2]], dtype=float),
        hz=10.0,
        renderer="meshcat",
        open_browser=True,
    )

    assert captured["opened"] == 1
