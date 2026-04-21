"""Backend selection helpers for rendering and kinematics."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping


@dataclass(slots=True)
class KinematicsSpec:
    """Canonical kinematics backend selection."""

    backend: str
    model_path: str | Path | None = None
    base_link: str | None = None
    end_link: str | None = None
    options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.backend = self.backend.strip().lower()
        if not self.backend:
            raise ValueError("backend must not be empty.")
        self.options = dict(self.options)

    def resolve(self, *, model_path: str | Path | None = None) -> tuple[str | Path, dict[str, Any]]:
        """Resolve the backend-specific model path and keyword arguments."""

        resolved_model_path = self.model_path if self.model_path is not None else model_path
        if resolved_model_path is None:
            raise ValueError("A kinematics model_path is required.")

        kwargs = dict(self.options)
        if self.base_link is not None:
            kwargs.setdefault("base_link", self.base_link)
        if self.end_link is not None:
            kwargs.setdefault("end_link", self.end_link)
        return resolved_model_path, kwargs


@dataclass(slots=True)
class RenderSpec:
    """Canonical renderer selection with optional kinematics wiring."""

    renderer: str = "matplotlib"
    options: dict[str, Any] = field(default_factory=dict)
    kinematics: KinematicsSpec | None = None

    def __post_init__(self) -> None:
        self.renderer = self.renderer.strip().lower()
        if not self.renderer:
            raise ValueError("renderer must not be empty.")
        self.options = dict(self.options)

    def with_overrides(
        self,
        *,
        options: Mapping[str, Any] | None = None,
        kinematics: KinematicsSpec | None = None,
    ) -> "RenderSpec":
        """Return a copy with merged backend options."""

        merged_options = dict(self.options)
        if options is not None:
            merged_options.update(options)
        return replace(
            self,
            options=merged_options,
            kinematics=self.kinematics if kinematics is None else kinematics,
        )

    def resolve_kwargs(self, *, model_path: str | Path | None = None) -> dict[str, Any]:
        """Resolve backend keyword arguments, including optional kinematics config."""

        kwargs = dict(self.options)
        if self.kinematics is None:
            return kwargs

        kinematics_model_path, kinematics_kwargs = self.kinematics.resolve(model_path=model_path)
        kwargs.setdefault("kinematics_backend", self.kinematics.backend)
        kwargs.setdefault("kinematics_model_path", kinematics_model_path)
        for key, value in kinematics_kwargs.items():
            kwargs.setdefault(key, value)
        return kwargs


def coerce_kinematics_spec(value: str | KinematicsSpec | None) -> KinematicsSpec | None:
    """Normalize strings into :class:`KinematicsSpec`."""

    if value is None:
        return None
    if isinstance(value, KinematicsSpec):
        return value
    if isinstance(value, str):
        return KinematicsSpec(value)
    raise TypeError(f"Unsupported kinematics specification: {type(value)!r}")


def coerce_render_spec(
    value: str | RenderSpec,
    *,
    options: Mapping[str, Any] | None = None,
    kinematics: str | KinematicsSpec | None = None,
) -> RenderSpec:
    """Normalize strings and overlay backend options."""

    resolved_kinematics = coerce_kinematics_spec(kinematics)
    if isinstance(value, RenderSpec):
        return value.with_overrides(options=options, kinematics=resolved_kinematics)
    if isinstance(value, str):
        return RenderSpec(renderer=value, options=dict(options or {}), kinematics=resolved_kinematics)
    raise TypeError(f"Unsupported renderer specification: {type(value)!r}")


__all__ = [
    "KinematicsSpec",
    "RenderSpec",
    "coerce_kinematics_spec",
    "coerce_render_spec",
]
