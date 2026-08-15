"""Unit canonicalization for commensurable pooling (gap #1, 2026-08-15).

A pool is only meaningful when inputs share a scale. Same-family units
(kbar/GPa/MPa/bar) convert exactly — multiplicative factors, temperature
affine — so convert to a family base BEFORE grouping. Never convert across
families; unknown/exotic units ('‰ VSMOW', 'fold', 'ratio') stay untouched
and pool separately, exactly as before.

Exact factors (no approximations):
  1 GPa = 10 kbar = 10⁴ bar; 1 MPa = 10⁻² kbar; 1 kPa = 10⁻⁵ kbar
  1 Ga = 10³ Ma; 1 ka = 10⁻³ Ma
  1 km = 10³ m = 10⁵ cm = 10⁶ mm; 1 µm = 10⁻⁹ km; 1 nm = 10⁻¹² km
  1 ‰ = 0.1 % (wt% treated as mass-% scale for pooling; noted at call site)
  1 ppb = 10⁻³ ppm
  K = °C + 273.15 (affine: value shifts, uncertainty unchanged)
"""

from __future__ import annotations

MULTIPLICATIVE: dict[str, dict[str, float]] = {
    "kbar": {"kbar": 1.0, "GPa": 10.0, "MPa": 0.01, "bar": 1e-3, "kPa": 1e-5},
    "Ma": {"Ma": 1.0, "Ga": 1000.0, "ka": 0.001},
    "km": {"km": 1.0, "m": 1e-3, "cm": 1e-5, "mm": 1e-6, "µm": 1e-9, "nm": 1e-12},
    "%": {"%": 1.0, "wt%": 1.0, "‰": 0.1},
    "ppm": {"ppm": 1.0, "ppb": 0.001},
}
AFFINE: dict[str, dict[str, tuple[float, float]]] = {
    "°C": {"°C": (1.0, 0.0), "K": (1.0, -273.15)},
}

# family-base unit → the measurement name used when a row's label is generic
BASE_MEASUREMENT: dict[str, str] = {
    "kbar": "pressure",
    "°C": "temperature",
    "Ma": "age",
    "km": "distance",
    "%": "percentage/composition",
    "ppm": "concentration",
}

_FAMILY_BY_UNIT: dict[str, str] = {}
for _base, _members in MULTIPLICATIVE.items():
    for _u in _members:
        _FAMILY_BY_UNIT[_u] = _base
for _base, _members in AFFINE.items():
    for _u in _members:
        _FAMILY_BY_UNIT[_u] = _base


def canonical_unit(unit: str) -> str | None:
    """Family base unit for `unit`, or None when not convertible."""
    return _FAMILY_BY_UNIT.get(unit)


def to_canonical(
    value: float, unit: str, uncertainty: float | None = None
) -> tuple[str, float, float | None] | None:
    """Convert (value, unit[, uncertainty]) to the family base.

    Returns (base_unit, value, uncertainty) or None when `unit` is not in a
    conversion family (caller keeps the row unchanged — never guesses).
    """
    base = _FAMILY_BY_UNIT.get(unit)
    if base is None:
        return None
    if base in MULTIPLICATIVE:
        f = MULTIPLICATIVE[base][unit]
        return base, value * f, None if uncertainty is None else uncertainty * f
    a, b = AFFINE[base][unit]
    return base, a * value + b, uncertainty
