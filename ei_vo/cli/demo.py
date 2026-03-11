"""Compatibility wrapper for the canonical playback CLI."""

from __future__ import annotations

from .playback import build_parser, build_trajectory, main


__all__ = ["build_parser", "build_trajectory", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
