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

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "models/three_dof_arm.urdf"
RENDERER = "matplotlib"  # Change to "meshcat" or "pyrender".
BACKEND = None  # Change to "literobo" when you need kinematics.
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
    elif RENDERER == "meshcat":
        renderer = RENDERER
        kwargs = {"hz": 240.0}
    elif RENDERER == "pyrender":
        renderer = "pyrender"
        kwargs = {
            "hz": 240.0,
            "record_path": ROOT.parent / "recordings" / "switch_renderer_pyrender.mp4",
        }
    else:
        raise ValueError(f"Unsupported renderer: {RENDERER!r}")

    render_program(
        MODEL,
        renderer=renderer,
        kinematics=backend,
        **kwargs,
    )


if __name__ == "__main__":
    main()
