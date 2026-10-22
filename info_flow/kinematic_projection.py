"""Dependency-free kinematic half-space projection used by planning/servo tests."""

import numpy as np


def project_spherical_command(u_info, a, c):
    """Project a two-component spherical command onto ``a.T @ u >= c``."""
    u = np.asarray(u_info, dtype=float)
    direction = np.asarray(a, dtype=float)
    if u.shape != direction.shape:
        raise ValueError("u_info and a must have the same shape")
    violation = float(c - np.dot(direction, u))
    if violation <= 0.0:
        return u.copy(), 0.0
    denom = float(np.dot(direction, direction))
    if denom <= 1e-12:
        raise ValueError("no usable kinematic correction direction")
    delta = (violation / denom) * direction
    return u + delta, float(np.linalg.norm(delta))
