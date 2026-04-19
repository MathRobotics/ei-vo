#!/usr/bin/env python3
"""Example: switch renderer and backend by editing variables."""

from __future__ import annotations

from pathlib import Path
import sys

if __package__ in {None, ""}:  # pragma: no cover - exercised by direct script execution.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from examples._bootstrap import ensure_repo_root_on_path
else:
    from ._bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()

from ei_vo import KinematicsSpec, RenderSpec, render_program
from ei_vo.mjpython import maybe_relaunch_with_mjpython

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "models/three_dof_arm.urdf"
RENDERER = "meshcat"  # Change to "mujoco", "matplotlib", or "blender".
BACKEND = "pinocchio"  # Change to "literobo" or None.
BASE_LINK = "base"
END_LINK = "ee"


def _resolve_backend() -> KinematicsSpec | None:
    if BACKEND is None:
        return None
    return KinematicsSpec(
        BACKEND,
        base_link=BASE_LINK,
        end_link=END_LINK,
    )


def main() -> None:
    backend = _resolve_backend()

    if RENDERER == "matplotlib":
        renderer = RenderSpec("matplotlib", options={"show": True, "title": "Renderer Switch"})
        kwargs = {"hz": 120.0}
    elif RENDERER in {"mujoco", "meshcat"}:
        renderer = RENDERER
        kwargs = {"program": "waypoints", "hz": 240.0}
    elif RENDERER == "blender":
        renderer = "blender"
        kwargs = {
            "program": "waypoints",
            "hz": 240.0,
            "record_path": ROOT.parent / "recordings" / "switch_renderer_blender.mp4",
        }
    else:
        raise ValueError(f"Unsupported renderer: {RENDERER!r}")

    maybe_relaunch_with_mjpython(RENDERER, exec_args=[__file__, *sys.argv[1:]])
    render_program(
        MODEL,
        renderer=renderer,
        kinematics=backend,
        **kwargs,
    )


if __name__ == "__main__":
    main()
