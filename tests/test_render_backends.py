import importlib
import sys
import pytest
import types

import numpy as np

from ei_vo import RenderSpec
from ei_vo.core import Trajectory


def _install_dummy_matplotlib(monkeypatch):
    captured = {}

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

        def legend(self, *args, **kwargs):
            captured["legend"] = True

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

    monkeypatch.setitem(sys.modules, "matplotlib", matplotlib)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", pyplot)
    return captured


def _install_dummy_meshcat(monkeypatch):
    captured = {
        "objects": {},
        "transforms": {},
        "opened": 0,
        "html": "<html>meshcat-scene</html>",
    }

    class DummyNode:
        def __init__(self, path=""):
            self.path = path

        def __getitem__(self, key):
            child_path = f"{self.path}/{key}" if self.path else key
            return DummyNode(child_path)

        def set_object(self, obj, material=None):
            captured["objects"][self.path] = (obj, material)

        def set_transform(self, transform):
            captured["transforms"].setdefault(self.path, []).append(np.asarray(transform))

        def open(self):
            captured["opened"] += 1
            return self

        def static_html(self):
            return captured["html"]

    class DummyVisualizer(DummyNode):
        pass

    class Sphere:
        def __init__(self, radius):
            self.radius = radius

    class Box:
        def __init__(self, lengths):
            self.lengths = lengths

    class Cylinder:
        def __init__(self, height, radius):
            self.height = height
            self.radius = radius

    class MeshPhongMaterial:
        def __init__(self, color, opacity=1.0, transparent=False):
            self.color = color
            self.opacity = opacity
            self.transparent = transparent

    geometry = types.ModuleType("meshcat.geometry")
    geometry.Sphere = Sphere
    geometry.Box = Box
    geometry.Cylinder = Cylinder
    geometry.MeshPhongMaterial = MeshPhongMaterial

    meshcat = types.ModuleType("meshcat")
    meshcat.Visualizer = DummyVisualizer
    meshcat.geometry = geometry

    monkeypatch.setitem(sys.modules, "meshcat", meshcat)
    monkeypatch.setitem(sys.modules, "meshcat.geometry", geometry)
    return captured


def test_render_package_is_lazy_without_mujoco():
    sys.modules.pop("ei_vo.render", None)
    sys.modules.pop("mujoco", None)
    sys.modules.pop("mujoco.viewer", None)

    render = importlib.import_module("ei_vo.render")

    assert render.available_renderers() == ("matplotlib", "meshcat", "mujoco")


def test_matplotlib_renderer_saves_image(monkeypatch, tmp_path, install_dummy_mujoco):
    install_dummy_mujoco()
    captured = _install_dummy_matplotlib(monkeypatch)
    sys.modules.pop("ei_vo.render.render_matplotlib", None)
    matplotlib_renderer = importlib.import_module("ei_vo.render.render_matplotlib")

    trajectory = Trajectory.from_positions(np.arange(12, dtype=float).reshape(4, 3), dt=0.1)
    matplotlib_renderer.play(
        "dummy.xml",
        trajectory,
        hz=10.0,
        show=False,
        record_path=tmp_path / "trajectory_plot",
    )

    assert captured["projection"] == "3d"
    assert len(captured["plots"]) == 1
    assert len(captured["scatters"]) == 1
    assert captured["savefig"][0].endswith("trajectory_plot.png")
    assert captured["xlabel"] == "x [m]"
    assert captured["ylabel"] == "y [m]"
    assert captured["zlabel"] == "z [m]"
    assert captured["title"].endswith("(4/4)")
    assert captured["box_aspect"] == (1.0, 1.0, 1.0)
    assert captured["closed"] is True


def test_matplotlib_renderer_animates_geometry_when_shown(monkeypatch, install_dummy_mujoco):
    install_dummy_mujoco()
    captured = _install_dummy_matplotlib(monkeypatch)
    sys.modules.pop("ei_vo.render.render_matplotlib", None)
    matplotlib_renderer = importlib.import_module("ei_vo.render.render_matplotlib")

    matplotlib_renderer.play(
        "dummy.xml",
        [[0.0, 0.2], [0.4, 0.6], [0.8, 1.0]],
        hz=4.0,
        show=True,
        title="Animated Geometry",
    )

    assert captured["shown"] == 1
    assert len(captured["pauses"]) == 2
    assert captured["clear_calls"] == 3
    assert len(captured["plots"]) == 3
    assert len(captured["scatters"]) == 3
    assert captured["title"] == "Animated Geometry (3/3)"


def test_generic_play_dispatches_matplotlib_renderer(monkeypatch, install_dummy_mujoco):
    install_dummy_mujoco()
    captured = _install_dummy_matplotlib(monkeypatch)
    sys.modules.pop("ei_vo.render.render_matplotlib", None)
    sys.modules.pop("ei_vo.render", None)
    render = importlib.import_module("ei_vo.render")

    render.play(
        "dummy.xml",
        [[0.0, 1.0], [1.0, 2.0]],
        hz=2.0,
        renderer="matplotlib",
        show=False,
        title="Angles",
    )

    assert captured["title"] == "Angles (2/2)"
    assert len(captured["plots"]) == 1
    assert len(captured["scatters"]) == 1


def test_generic_play_accepts_render_spec(monkeypatch, install_dummy_mujoco):
    install_dummy_mujoco()
    captured = _install_dummy_matplotlib(monkeypatch)
    sys.modules.pop("ei_vo.render.render_matplotlib", None)
    sys.modules.pop("ei_vo.render", None)
    render = importlib.import_module("ei_vo.render")

    render.play(
        "dummy.xml",
        [[0.0, 1.0], [1.0, 2.0]],
        hz=2.0,
        renderer=RenderSpec("plot", options={"show": False, "title": "Configured Plot"}),
    )

    assert captured["title"] == "Configured Plot (2/2)"
    assert len(captured["plots"]) == 1
    assert len(captured["scatters"]) == 1


def test_matplotlib_renderer_requires_model(monkeypatch):
    _install_dummy_matplotlib(monkeypatch)
    sys.modules.pop("ei_vo.render.render_matplotlib", None)
    matplotlib_renderer = importlib.import_module("ei_vo.render.render_matplotlib")

    with pytest.raises(ValueError, match="--model is required"):
        matplotlib_renderer.play(None, [[0.0, 0.0], [1.0, 1.0]], hz=2.0)


def test_meshcat_renderer_exports_html(monkeypatch, tmp_path, install_dummy_mujoco):
    install_dummy_mujoco()
    captured = _install_dummy_meshcat(monkeypatch)
    sys.modules.pop("ei_vo.render.render_meshcat", None)
    meshcat_renderer = importlib.import_module("ei_vo.render.render_meshcat")
    monkeypatch.setattr(meshcat_renderer.time, "sleep", lambda _: None)

    trajectory = Trajectory.from_positions(np.linspace(0.0, 1.0, 14, dtype=float).reshape(2, 7))
    output_path = tmp_path / "scene"
    meshcat_renderer.play(
        "dummy.xml",
        trajectory,
        hz=20.0,
        open_browser=False,
        record_path=output_path,
    )

    assert "geoms/0/shape" in captured["objects"]
    assert "geoms/1/shape" in captured["objects"]
    assert "geoms/2/cylinder" in captured["objects"]
    assert len(captured["transforms"]["geoms/0"]) == trajectory.steps
    assert output_path.with_suffix(".html").read_text(encoding="utf-8") == captured["html"]


def test_meshcat_material_uses_builtin_bool(monkeypatch, install_dummy_mujoco):
    install_dummy_mujoco()
    _install_dummy_meshcat(monkeypatch)
    sys.modules.pop("ei_vo.render.render_meshcat", None)
    meshcat_renderer = importlib.import_module("ei_vo.render.render_meshcat")

    class DummyMaterial:
        def __init__(self, color, opacity=1.0, transparent=False):
            self.color = color
            self.opacity = opacity
            self.transparent = transparent

    geometry = types.SimpleNamespace(MeshPhongMaterial=DummyMaterial)
    material = meshcat_renderer._rgba_to_material(geometry, np.array([0.1, 0.2, 0.3, 0.5]))

    assert isinstance(material.transparent, bool)
    assert material.transparent is True
    assert isinstance(material.opacity, float)


def test_generic_play_dispatches_meshcat_renderer(monkeypatch, install_dummy_mujoco):
    install_dummy_mujoco()
    captured = _install_dummy_meshcat(monkeypatch)
    sys.modules.pop("ei_vo.render.render_meshcat", None)
    sys.modules.pop("ei_vo.render", None)
    render = importlib.import_module("ei_vo.render")
    meshcat_renderer = importlib.import_module("ei_vo.render.render_meshcat")
    monkeypatch.setattr(meshcat_renderer.time, "sleep", lambda _: None)

    render.play(
        "dummy.xml",
        np.linspace(0.0, 1.0, 14, dtype=float).reshape(2, 7),
        hz=10.0,
        renderer="meshcat",
        open_browser=True,
    )

    assert captured["opened"] == 1
