"""CLI entrypoints for ``ei_vo``."""

from .playback import build_parser, build_trajectory, main

__all__ = ["build_parser", "build_trajectory", "main"]
