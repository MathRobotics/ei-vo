import importlib
import json
import pathlib

import numpy as np


def _threejs_matrix(matrix: np.ndarray) -> list[float]:
    return np.asarray(matrix, dtype=float).T.reshape(-1).tolist()


def test_load_camera_settings_from_preset_json(tmp_path: pathlib.Path):
    config = importlib.import_module("ei_vo.config")
    camera_path = tmp_path / "front.camera.json"
    camera_path.write_text(
        json.dumps(
            {
                "distance": 4.0,
                "azimuth": 15.0,
                "elevation": -20.0,
                "lookat": [0.1, 0.2, 0.3],
            }
        ),
        encoding="utf-8",
    )

    settings = config.load_camera_settings(camera_path)

    assert settings.distance == 4.0
    assert settings.azimuth == 15.0
    assert settings.elevation == -20.0
    np.testing.assert_allclose(settings.lookat, [0.1, 0.2, 0.3])


def test_coerce_camera_settings_accepts_json_path(tmp_path: pathlib.Path):
    config = importlib.import_module("ei_vo.config")
    camera_path = tmp_path / "front.camera.json"
    camera_path.write_text(json.dumps({"distance": 2.0}), encoding="utf-8")

    settings = config.coerce_camera_settings(camera_path)

    assert settings is not None
    assert settings.distance == 2.0


def test_load_camera_settings_from_meshcat_scene_json(tmp_path: pathlib.Path):
    config = importlib.import_module("ei_vo.config")
    scene_path = tmp_path / "scene.json"

    lookat_transform = np.eye(4, dtype=float)
    lookat_transform[:3, 3] = np.array([0.5, -0.25, 1.2], dtype=float)
    rotated_transform = np.eye(4, dtype=float)
    rotated_transform[:3, :3] = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=float,
    )

    scene_path.write_text(
        json.dumps(
            {
                "object": {
                    "name": "Scene",
                    "type": "Scene",
                    "children": [
                        {
                            "name": "Cameras",
                            "type": "Object3D",
                            "children": [
                                {
                                    "name": "default",
                                    "type": "Object3D",
                                    "matrix": _threejs_matrix(lookat_transform),
                                    "children": [
                                        {
                                            "name": "rotated",
                                            "type": "Object3D",
                                            "matrix": _threejs_matrix(rotated_transform),
                                            "children": [
                                                {
                                                    "name": "<object>",
                                                    "type": "PerspectiveCamera",
                                                    "position": [-2.0, 0.0, 0.0],
                                                }
                                            ],
                                        }
                                    ],
                                }
                            ],
                        }
                    ],
                }
            }
        ),
        encoding="utf-8",
    )

    settings = config.load_camera_settings(scene_path)

    assert settings.distance == 2.0
    assert settings.azimuth == 0.0
    assert settings.elevation == 0.0
    np.testing.assert_allclose(settings.lookat, [0.5, -0.25, 1.2])


def test_save_camera_settings_writes_portable_json(tmp_path: pathlib.Path):
    config = importlib.import_module("ei_vo.config")
    output_path = tmp_path / "saved.camera.json"

    saved_path = config.save_camera_settings(
        {
            "distance": 3.0,
            "azimuth": 45.0,
            "elevation": 10.0,
            "lookat": (1.0, 2.0, 3.0),
        },
        output_path,
    )

    assert saved_path == output_path
    assert json.loads(output_path.read_text(encoding="utf-8")) == {
        "distance": 3.0,
        "azimuth": 45.0,
        "elevation": 10.0,
        "lookat": [1.0, 2.0, 3.0],
    }
