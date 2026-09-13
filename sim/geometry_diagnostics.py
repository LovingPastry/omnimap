"""Geometry diagnostics for the hemisphere control loop.

These helpers exist to make the spherical-parameterization degeneracy measurable.
The theta degree of freedom collapses as cos(phi) twice over -- once in the Fisher
probe arclength, once in the ``e_theta`` basis norm -- so raw ``grad_theta`` and raw
``theta_rate`` are not comparable across elevations. The quantities here are the
ones that *are* comparable: normalized-basis velocity components, great-circle
arclength, and longitude coverage.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np

LONGITUDE_BIN_DEG = 20.0
NUM_LONGITUDE_BINS = int(round(360.0 / LONGITUDE_BIN_DEG))


def longitude_bin(theta: float) -> int:
    """Bin azimuth into 20-degree sectors, so coverage is countable out of 18."""
    theta_deg = math.degrees(float(theta)) % 360.0
    return int(theta_deg // LONGITUDE_BIN_DEG)


def wrap_angle_signed(delta: float) -> float:
    """Wrap an angle difference into (-pi, pi] so theta steps never read as ~2pi."""
    return float((float(delta) + math.pi) % (2.0 * math.pi) - math.pi)


def unit_normal(position: Sequence[float], center: Sequence[float]) -> np.ndarray:
    offset = np.asarray(position, dtype=np.float64).reshape(3) - np.asarray(
        center, dtype=np.float64
    ).reshape(3)
    radius = float(np.linalg.norm(offset))
    if radius < 1e-12:
        return np.zeros(3, dtype=np.float64)
    return offset / radius


def great_circle_arclen(
    prev_position: Optional[Sequence[float]],
    position: Sequence[float],
    center: Sequence[float],
) -> float:
    """Arclength along the sphere, not the chord.

    Chord length understates motion when the step is large, which is exactly the
    regime a polar oscillation produces.
    """
    if prev_position is None:
        return 0.0
    n_prev = unit_normal(prev_position, center)
    n_cur = unit_normal(position, center)
    if not np.any(n_prev) or not np.any(n_cur):
        return 0.0
    radius = float(
        np.linalg.norm(
            np.asarray(position, dtype=np.float64).reshape(3)
            - np.asarray(center, dtype=np.float64).reshape(3)
        )
    )
    cos_angle = float(np.clip(np.dot(n_prev, n_cur), -1.0, 1.0))
    return radius * math.acos(cos_angle)


def normalized_spherical_basis(
    theta: float, phi: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unit-length e_theta / e_phi / n_hat.

    ``e_theta`` is divided by cos(phi) so it stays unit length. This is the whole
    point: projecting velocity onto the *normalized* basis measures how fast the
    camera actually moves east, rather than re-introducing the cos(phi) that the
    unnormalized basis carries.
    """
    ct, st = math.cos(theta), math.sin(theta)
    cp, sp = math.cos(phi), math.sin(phi)
    n_hat = np.array([cp * ct, cp * st, sp], dtype=np.float64)
    e_phi = np.array([-sp * ct, -sp * st, cp], dtype=np.float64)
    if abs(cp) < 1e-9:
        # At the pole the east direction is undefined; report zero rather than a
        # numerically amplified garbage direction.
        e_theta_hat = np.zeros(3, dtype=np.float64)
    else:
        e_theta_hat = np.array([-st, ct, 0.0], dtype=np.float64)
    return e_theta_hat, e_phi, n_hat


def velocity_components(
    velocity: Sequence[float], theta: float, phi: float
) -> dict[str, float]:
    """Decompose a world velocity onto the unit spherical basis."""
    v = np.asarray(velocity, dtype=np.float64).reshape(3)
    e_theta_hat, e_phi, n_hat = normalized_spherical_basis(theta, phi)
    v_radial = float(np.dot(v, n_hat))
    v_tangential = v - v_radial * n_hat
    return {
        "v_along_e_theta_hat": float(np.dot(v, e_theta_hat)),
        "v_along_e_phi_hat": float(np.dot(v, e_phi)),
        "v_tangential_norm": float(np.linalg.norm(v_tangential)),
        "v_radial_signed": v_radial,
    }


def sec_phi_compensated(grad_theta: float, phi: float, *, cap: float = 1e6) -> float:
    """``grad_theta / cos(phi)`` -- the flatness check.

    Restores the single cos(phi) that the angle-space probe loses, so the result
    should be roughly phi-independent. Capped because sec(phi) diverges at the pole.
    """
    cp = math.cos(float(phi))
    if abs(cp) < 1e-9:
        return float("nan")
    value = float(grad_theta) / cp
    if not math.isfinite(value) or abs(value) > cap:
        return float("nan")
    return value


class GeometryDiagnostics:
    """Per-step geometry metrics that need memory of the previous step."""

    def __init__(self) -> None:
        self._prev_theta: Optional[float] = None
        self._prev_position: Optional[np.ndarray] = None
        self._prev_d_theta: Optional[float] = None
        self.visited_longitude_bins: set[int] = set()
        self.theta_reversal_count = 0
        self.pole_crossing_count = 0
        self.total_arclen_m = 0.0

    def update(
        self,
        *,
        theta: float,
        phi: float,
        position: Sequence[float],
        center: Sequence[float],
        velocity: Optional[Sequence[float]] = None,
        grad_theta_raw: Optional[float] = None,
    ) -> dict[str, object]:
        position = np.asarray(position, dtype=np.float64).reshape(3)

        d_theta = (
            wrap_angle_signed(theta - self._prev_theta)
            if self._prev_theta is not None
            else 0.0
        )
        # A reversal is a sign flip of the azimuth step. Repeated reversals while
        # pinned at high phi is the actual pathology; a single pole transit is not.
        reversal = bool(
            self._prev_d_theta is not None
            and abs(d_theta) > 1e-9
            and abs(self._prev_d_theta) > 1e-9
            and (d_theta > 0.0) != (self._prev_d_theta > 0.0)
        )
        # theta jumps ~180 deg when a great circle passes over the pole. This is a
        # chart artifact, not a fault -- read it together with longitude coverage.
        crossing = bool(abs(d_theta) > math.radians(90.0))
        arclen = great_circle_arclen(self._prev_position, position, center)

        if reversal:
            self.theta_reversal_count += 1
        if crossing:
            self.pole_crossing_count += 1
        self.total_arclen_m += arclen
        bin_idx = longitude_bin(theta)
        self.visited_longitude_bins.add(bin_idx)

        row: dict[str, object] = {
            "longitude_bin": bin_idx,
            "d_theta_wrapped": d_theta,
            "theta_reversal": reversal,
            "pole_crossing": crossing,
            "arclen_step_m": arclen,
            "longitude_bins_visited": len(self.visited_longitude_bins),
            "theta_reversal_count": self.theta_reversal_count,
            "pole_crossing_count": self.pole_crossing_count,
            "path_arclen_m": self.total_arclen_m,
        }
        if velocity is not None:
            row.update(velocity_components(velocity, theta, phi))
        if grad_theta_raw is not None:
            row["grad_theta_sec_phi"] = sec_phi_compensated(grad_theta_raw, phi)

        self._prev_theta = float(theta)
        self._prev_position = position.copy()
        if abs(d_theta) > 1e-9:
            self._prev_d_theta = d_theta
        return row

    def summary(self, center: Sequence[float], position: Sequence[float]) -> dict:
        return {
            "longitude_bins_visited": len(self.visited_longitude_bins),
            "longitude_bins_total": NUM_LONGITUDE_BINS,
            "theta_reversal_count": self.theta_reversal_count,
            "pole_crossing_count": self.pole_crossing_count,
            "path_arclen_m": self.total_arclen_m,
        }


GEOMETRY_FIELD_NAMES = [
    "longitude_bin",
    "d_theta_wrapped",
    "theta_reversal",
    "pole_crossing",
    "arclen_step_m",
    "longitude_bins_visited",
    "theta_reversal_count",
    "pole_crossing_count",
    "path_arclen_m",
    "v_along_e_theta_hat",
    "v_along_e_phi_hat",
    "v_tangential_norm",
    "v_radial_signed",
    "grad_theta_sec_phi",
]
