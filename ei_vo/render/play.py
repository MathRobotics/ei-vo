"""Generic renderer dispatch preserving the package-level ``play`` API."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

from ..backends import KinematicsSpec, RenderSpec, coerce_render_spec
from .registry import render as dispatch_render


def play(
    model_path: str | None,
    traj,
    slow: float = 1.0,
    hz: float = 240.0,
    camera=None,
    loop: bool = False,
    record_path: Optional[str] = None,
    record_fps: Optional[float] = None,
    record_size: Optional[Tuple[int, int]] = None,
    record_frames_dir: str | Path | None = None,
    renderer: str | RenderSpec = "mujoco",
    kinematics: str | KinematicsSpec | None = None,
    **backend_kwargs,
):
    if record_frames_dir is not None and record_path is None:
        raise ValueError("record_frames_dir requires record_path.")
    render_spec = coerce_render_spec(
        renderer,
        options=backend_kwargs,
        kinematics=kinematics,
    )
    return dispatch_render(
        render_spec.renderer,
        model_path=model_path,
        traj=traj,
        slow=slow,
        hz=hz,
        camera=camera,
        loop=loop,
        record_path=record_path,
        record_fps=record_fps,
        record_size=record_size,
        record_frames_dir=record_frames_dir,
        **render_spec.resolve_kwargs(model_path=model_path),
    )
