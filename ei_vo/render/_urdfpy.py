"""Shared NumPy and urdfpy compatibility helpers for renderer backends."""

from __future__ import annotations

import collections
import collections.abc
import fractions
import math
import re
import sys
import types

import numpy as np


def _parse_version_fallback(value: object):
    text = str(value)
    parts: list[tuple[int, object]] = []
    for part in re.split(r"[^0-9A-Za-z]+", text):
        if not part:
            continue
        if part.isdigit():
            parts.append((0, int(part)))
        else:
            parts.append((1, part.lower()))
    return tuple(parts)


def install_urdfpy_compat_shims() -> None:
    """Patch stdlib and NumPy symbols needed by older urdfpy / pyrender stacks."""

    for name in ("Iterable", "Mapping", "MutableMapping", "Sequence", "Set"):
        if not hasattr(collections, name):
            setattr(collections, name, getattr(collections.abc, name))

    if not hasattr(fractions, "gcd"):
        fractions.gcd = math.gcd  # type: ignore[attr-defined]

    numpy_aliases = {
        "Inf": np.inf,
        "Infinity": np.inf,
        "NaN": np.nan,
        "bool": bool,
        "complex_": np.complex128,
        "float": float,
        "float_": np.float64,
        "int": int,
        "int_": np.int64,
        "infty": np.inf,
        "object": object,
        "unicode_": np.str_,
    }
    for name, value in numpy_aliases.items():
        if name not in np.__dict__:
            setattr(np, name, value)

    if "pkg_resources" not in sys.modules:
        pkg_resources = types.ModuleType("pkg_resources")

        def parse_version(value: object):
            try:
                from packaging.version import parse as packaging_parse_version
            except Exception:
                return _parse_version_fallback(value)
            return packaging_parse_version(str(value))

        pkg_resources.parse_version = parse_version  # type: ignore[attr-defined]
        sys.modules["pkg_resources"] = pkg_resources


__all__ = ["install_urdfpy_compat_shims"]
