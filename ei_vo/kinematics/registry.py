"""Kinematics backend registration and lazy loading."""

from __future__ import annotations

import importlib
from types import ModuleType

_BACKEND_MODULES: dict[str, ModuleType] = {}
_BUILTIN_BACKENDS: dict[str, str] = {
    "literobo": "ei_vo.kinematics.literobo_backend",
}


def register_kinematics_backend(name: str, module: ModuleType, *, replace: bool = False) -> None:
    """Register a kinematics backend module."""

    normalized = name.strip().lower()
    if not normalized:
        raise ValueError("Kinematics backend name must not be empty.")
    if normalized in _BACKEND_MODULES and not replace:
        raise ValueError(f"Kinematics backend {normalized!r} is already registered.")
    _BACKEND_MODULES[normalized] = module


def _load_builtin_backend(name: str) -> ModuleType:
    target = _BUILTIN_BACKENDS.get(name)
    if target is None:
        raise ValueError(
            f"Unknown kinematics backend {name!r}. "
            f"Available backends: {', '.join(available_kinematics_backends())}"
        )
    module = importlib.import_module(target)
    register_kinematics_backend(name, module, replace=True)
    return module


def get_kinematics_backend(name: str) -> ModuleType:
    """Return a registered backend module, loading built-ins lazily."""

    normalized = name.strip().lower()
    module = _BACKEND_MODULES.get(normalized)
    if module is not None:
        return module
    return _load_builtin_backend(normalized)


def available_kinematics_backends() -> tuple[str, ...]:
    """List built-in and user-registered kinematics backends."""

    return tuple(sorted(set(_BUILTIN_BACKENDS) | set(_BACKEND_MODULES)))


def load_model_dof(
    backend: str,
    model_path,
    /,
    **kwargs,
) -> int:
    """Load model DOF using the selected backend."""

    module = get_kinematics_backend(backend)
    return int(module.load_model_dof(model_path, **kwargs))


def forward_kinematics(
    backend: str,
    model_path,
    traj,
    /,
    **kwargs,
):
    """Run forward kinematics using the selected backend."""

    module = get_kinematics_backend(backend)
    return module.forward_kinematics(model_path, traj, **kwargs)


__all__ = [
    "available_kinematics_backends",
    "forward_kinematics",
    "get_kinematics_backend",
    "load_model_dof",
    "register_kinematics_backend",
]
