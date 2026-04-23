"""Renderer package exports with lazy backend loading."""

from __future__ import annotations

import importlib

_LAZY_EXPORTS = {
    "CameraSettings": ("..config", "CameraSettings"),
    "PlaybackConfig": ("..config", "PlaybackConfig"),
    "RecordingConfig": ("..config", "RecordingConfig"),
    "available_renderers": (".registry", "available_renderers"),
    "get_renderer": (".registry", "get_renderer"),
    "play": (".play", "play"),
    "play_trajectory": (".play", "play"),
    "register_renderer": (".registry", "register_renderer"),
}

__all__ = [
    "CameraSettings",
    "PlaybackConfig",
    "RecordingConfig",
    "available_renderers",
    "get_renderer",
    "play",
    "play_trajectory",
    "register_renderer",
]


def __getattr__(name: str):
    try:
        module_name, attr_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    value = getattr(importlib.import_module(module_name, __name__), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__) | set(_LAZY_EXPORTS))
