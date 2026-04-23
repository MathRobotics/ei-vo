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

    class DummyBox:
        def __init__(self, lengths):
            self.lengths = np.asarray(lengths, dtype=float)

    class DummyCylinder:
        def __init__(self, height, radius=1.0, radiusTop=None, radiusBottom=None):
            self.height = float(height)
            self.radius = float(radius)
            self.radiusTop = None if radiusTop is None else float(radiusTop)
            self.radiusBottom = None if radiusBottom is None else float(radiusBottom)

    class DummySphere:
        def __init__(self, radius):
            self.radius = float(radius)

    class DummyTriangularMeshGeometry:
        def __init__(self, vertices, faces, color=None):
            self.vertices = np.asarray(vertices, dtype=float)
            self.faces = np.asarray(faces, dtype=np.uint32)
            self.color = None if color is None else np.asarray(color, dtype=float)

    class DummyMeshLambertMaterial:
        def __init__(self, **kwargs):
            self.kwargs = dict(kwargs)

    geometry = types.ModuleType("meshcat.geometry")
    geometry.Box = DummyBox
    geometry.Cylinder = DummyCylinder
    geometry.Sphere = DummySphere
    geometry.TriangularMeshGeometry = DummyTriangularMeshGeometry
    geometry.MeshLambertMaterial = DummyMeshLambertMaterial

    meshcat = types.ModuleType("meshcat")
    meshcat.__path__ = []
    meshcat.Visualizer = DummyVisualizer
    meshcat.animation = animation
    meshcat.geometry = geometry

    monkeypatch.setitem(sys.modules, "meshcat", meshcat)
    monkeypatch.setitem(sys.modules, "meshcat.animation", animation)
    monkeypatch.setitem(sys.modules, "meshcat.geometry", geometry)
    return captured


def _install_dummy_urdfpy_meshcat(monkeypatch):
    captured = {"configs": []}

    class DummyMesh:
        def __init__(self, vertices, faces):
            self.vertices = np.asarray(vertices, dtype=float)
            self.faces = np.asarray(faces, dtype=np.uint32)

    class DummyMaterial:
        def __init__(self, color):
            self.color = np.asarray(color, dtype=float)

    class DummyBoxGeometry:
        def __init__(self, size):
            self.size = np.asarray(size, dtype=float)

    class DummyCylinderGeometry:
        def __init__(self, radius, length):
            self.radius = float(radius)
            self.length = float(length)

    class DummyMeshGeometry:
        def __init__(self, meshes, scale=None):
            self.meshes = list(meshes)
            self.scale = None if scale is None else np.asarray(scale, dtype=float)

    class DummyGeometry:
        def __init__(self, *, box=None, cylinder=None, sphere=None, mesh=None):
            self.box = box
            self.cylinder = cylinder
            self.sphere = sphere
            self.mesh = mesh

    class DummyVisual:
        def __init__(self, geometry, origin=None, material=None):
            self.geometry = geometry
            self.origin = np.eye(4, dtype=float) if origin is None else np.asarray(origin, dtype=float)
            self.material = material

    class DummyLink:
        def __init__(self, name, visuals):
            self.name = name
            self.visuals = list(visuals)

    class DummySphereGeometry:
        def __init__(self, radius):
            self.radius = float(radius)

    class DummyRobot:
        def __init__(self):
            base_origin = np.eye(4, dtype=float)
            base_origin[:3, 3] = np.array([0.0, 0.0, 0.05], dtype=float)
            link_origin = np.eye(4, dtype=float)
            link_origin[:3, 3] = np.array([0.0, 0.0, 0.15], dtype=float)

            mesh = DummyMesh(
                vertices=[
                    [-0.02, -0.02, 0.0],
                    [0.02, -0.02, 0.0],
                    [0.0, 0.02, 0.0],
                ],
                faces=[[0, 1, 2]],
            )

            self.base = DummyLink(
                "base",
                [
                    DummyVisual(
                        DummyGeometry(box=DummyBoxGeometry([0.2, 0.2, 0.1])),
                        origin=base_origin,
                        material=DummyMaterial([0.8, 0.2, 0.2, 1.0]),
                    )
                ],
            )
            self.link1 = DummyLink(
                "link1",
                [
                    DummyVisual(
                        DummyGeometry(cylinder=DummyCylinderGeometry(0.03, 0.3)),
                        origin=link_origin,
                        material=DummyMaterial([0.2, 0.4, 0.8, 1.0]),
                    )
                ],
            )
            self.ee = DummyLink(
                "ee",
                [
                    DummyVisual(
                        DummyGeometry(
                            mesh=DummyMeshGeometry([mesh], scale=[1.0, 1.0, 1.0]),
                        ),
                        material=DummyMaterial([0.2, 0.8, 0.4, 0.6]),
                    ),
                    DummyVisual(
                        DummyGeometry(sphere=DummySphereGeometry(0.02)),
                        material=DummyMaterial([0.9, 0.9, 0.1, 1.0]),
                    ),
                ],
            )
            self.links = [self.base, self.link1, self.ee]

        def link_fk(self, cfg=None):
            config = {key: float(value) for key, value in dict(cfg or {}).items()}
            captured["configs"].append(config)
            joint1 = float(config.get("joint1", 0.0))
            joint2 = float(config.get("joint2", 0.0))

            base_pose = np.eye(4, dtype=float)
            link1_pose = np.eye(4, dtype=float)
            link1_pose[0, 3] = joint1
            ee_pose = np.eye(4, dtype=float)
            ee_pose[0, 3] = joint1 + joint2

            return {
                self.base: base_pose,
                self.link1: link1_pose,
                self.ee: ee_pose,
            }

    class DummyURDF:
        @staticmethod
        def load(path):
            captured["loaded_path"] = path
            return DummyRobot()

    urdfpy = types.ModuleType("urdfpy")
    urdfpy.URDF = DummyURDF

    monkeypatch.setitem(sys.modules, "urdfpy", urdfpy)
    return captured


def test_render_package_is_lazy_without_optional_renderers():
    sys.modules.pop("ei_vo.render", None)
    sys.modules.pop("meshcat", None)
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
    urdfpy_captured = _install_dummy_urdfpy_meshcat(monkeypatch)
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

    assert urdfpy_captured["loaded_path"] == model_path.resolve().as_posix()
    assert urdfpy_captured["configs"] == [
        {"joint1": 1.0, "joint2": -0.5},
        {"joint1": -1.0, "joint2": 0.5},
    ]
    assert "robot/visuals/base" in captured["objects"]
    assert "robot/visuals/link1" in captured["objects"]
    assert "robot/visuals/ee" in captured["objects"]
    assert "robot/visuals/ee_1" in captured["objects"]
    assert captured["properties"][("robot/collisions", "visible")] is False
    assert captured["properties"][("robot/visuals", "visible")] is True
    assert len(captured["transforms"]["robot/visuals/ee"]) == trajectory.steps
    assert len(captured["animations"]) == 1
    assert len(captured["animations"][0]["animation"].frames["robot/visuals/ee"]) == trajectory.steps
    np.testing.assert_allclose(captured["transforms"]["robot/visuals/ee"][0][:3, 3], [0.5, 0.0, 0.0])
    assert output_path.with_suffix(".html").read_text(encoding="utf-8") == captured["html"]
    assert opened_html == [output_path.with_suffix(".html")]


def test_meshcat_renderer_applies_lookat_to_urdf_visualizer(monkeypatch, tmp_path):
    captured = _install_dummy_meshcat(monkeypatch)
    _install_dummy_urdfpy_meshcat(monkeypatch)
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


def test_meshcat_renderer_can_hold_browser_open(monkeypatch, tmp_path):
    captured = _install_dummy_meshcat(monkeypatch)
    _install_dummy_urdfpy_meshcat(monkeypatch)
    sys.modules.pop("ei_vo.render.render_meshcat", None)
    meshcat_renderer = importlib.import_module("ei_vo.render.render_meshcat")
    monkeypatch.setattr(meshcat_renderer.time, "sleep", lambda _: None)
    hold_calls = []
    monkeypatch.setattr(meshcat_renderer, "_wait_until_interrupted", lambda interval_s=0.5: hold_calls.append(interval_s))

    model_path = _write_demo_urdf(tmp_path / "robot.urdf")
    meshcat_renderer.play(
        model_path,
        Trajectory.from_positions(np.zeros((1, 2), dtype=float), dt=0.1),
        hz=20.0,
        open_browser=True,
        hold_open=True,
    )

    assert captured["opened"] == 1
    assert hold_calls == [0.5]


def test_meshcat_renderer_rejects_record_size(monkeypatch, tmp_path):
    _install_dummy_meshcat(monkeypatch)
    _install_dummy_urdfpy_meshcat(monkeypatch)
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
    _install_dummy_urdfpy_meshcat(monkeypatch)
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
    _install_dummy_urdfpy_meshcat(monkeypatch)
    sys.modules.pop("ei_vo.render.render_meshcat", None)
    meshcat_renderer = importlib.import_module("ei_vo.render.render_meshcat")
    model_path = tmp_path / "robot.xml"
    model_path.write_text("<robot/>", encoding="utf-8")

    with pytest.raises(ValueError, match="only supports URDF"):
        meshcat_renderer.play(model_path, Trajectory.from_positions(np.zeros((1, 2), dtype=float), dt=0.1))


def test_generic_play_dispatches_meshcat_renderer(monkeypatch, tmp_path):
    captured = _install_dummy_meshcat(monkeypatch)
    _install_dummy_urdfpy_meshcat(monkeypatch)
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
