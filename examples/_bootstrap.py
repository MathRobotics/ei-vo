"""Helpers for running example scripts directly from the repository."""

from __future__ import annotations

import sys
from pathlib import Path

EXAMPLES_ROOT = Path(__file__).resolve().parent
REPO_ROOT = EXAMPLES_ROOT.parent


def ensure_repo_root_on_path() -> Path:
    """Prepend the repository root so local imports win over site-packages."""

    repo_root = REPO_ROOT.as_posix()
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    return REPO_ROOT


__all__ = ["EXAMPLES_ROOT", "REPO_ROOT", "ensure_repo_root_on_path"]
