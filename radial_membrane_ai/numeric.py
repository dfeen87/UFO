"""Small numeric trust-boundary helpers used by governed public operators."""

from __future__ import annotations

import math
from typing import Any


def finite_real(value: Any) -> float | None:
    """Return a finite binary float for an ``int``/``float``, else ``None``.

    Python integers are unbounded, while the equations in this package execute
    in finite floating-point arithmetic.  Conversion is therefore part of the
    trust boundary and may itself fail for hostile, arbitrarily large integers.
    Strings and booleans are intentionally not coerced.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        converted = float(value)
    except (OverflowError, TypeError, ValueError):
        return None
    return converted if math.isfinite(converted) else None


def require_finite_real(value: Any, name: str) -> float:
    """Validate and return a finite model scalar."""
    converted = finite_real(value)
    if converted is None:
        raise ValueError(f"{name} must be a finite real number.")
    return converted
