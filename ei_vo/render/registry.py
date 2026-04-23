"""Renderer backend registration and lazy loading."""

from __future__ import annotations

import importlib
from collections.abc import Callable

Renderer = Callable[..., object]

_RENDERERS: dict[str, Renderer] = {}
_BUILTIN_BACKENDS: dict[str, str] = {
    "matplotlib": "ei_vo.render.render_matplotlib:play",
    "meshcat": "ei_vo.render.render_meshcat:play",
    "pyrender": "ei_vo.render.render_pyrender:play",
}
_RENDERER_ALIASES: dict[str, str] = {
    "plot": "matplotlib",
}


def _normalize_renderer_name(name: str) -> str:
    normalized = name.strip().lower()
    if not normalized:
        raise ValueError("Renderer name must not be empty.")
    return _RENDERER_ALIASES.get(normalized, normalized)


def register_renderer(name: str, renderer: Renderer, *, replace: bool = False) -> None:
    """Register a renderer backend."""

    normalized = _normalize_renderer_name(name)
    if normalized in _RENDERERS and not replace:
        raise ValueError(f"Renderer {normalized!r} is already registered.")
    _RENDERERS[normalized] = renderer


def _load_builtin_renderer(name: str) -> Renderer:
    target = _BUILTIN_BACKENDS.get(name)
    if target is None:
        raise ValueError(
            f"Unknown renderer {name!r}. Available renderers: {', '.join(available_renderers())}"
        )
    module_name, function_name = target.split(":", 1)
    module = importlib.import_module(module_name)
    renderer = getattr(module, function_name)
    register_renderer(name, renderer, replace=True)
    return renderer


def get_renderer(name: str) -> Renderer:
    """Return a registered renderer, loading built-ins lazily."""

    normalized = _normalize_renderer_name(name)
    renderer = _RENDERERS.get(normalized)
    if renderer is not None:
        return renderer
    return _load_builtin_renderer(normalized)


def available_renderers() -> tuple[str, ...]:
    """List built-in and user-registered renderer names."""

    return tuple(sorted(set(_BUILTIN_BACKENDS) | set(_RENDERERS)))


def render(renderer: str, /, **kwargs):
    """Dispatch a render request to the selected backend."""

    backend = get_renderer(renderer)
    return backend(**kwargs)


__all__ = [
    "available_renderers",
    "get_renderer",
    "register_renderer",
    "render",
]
