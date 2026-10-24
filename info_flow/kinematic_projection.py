"""Pure NumPy half-space projection of two spherical rates."""

import numpy as np


def project_spherical_command(u_info, a, c):
    """Project a two-component spherical command onto ``a.T @ u >= c``."""
    u = np.asarray(u_info, dtype=float)
    direction = np.asarray(a, dtype=float)
    if u.shape != (2,) or direction.shape != (2,):
        raise ValueError("u_info and a must each contain two spherical rates")
    bound = np.asarray(c, dtype=float)
    if bound.shape != () or not np.isfinite(bound):
        raise ValueError("c must be a finite scalar")
    if not np.all(np.isfinite(u)) or not np.all(np.isfinite(direction)):
        raise ValueError("u_info and a must be finite")
    violation = float(bound - np.dot(direction, u))
    if not np.isfinite(violation):
        raise ValueError("kinematic constraint is not finite")
    if violation <= 0.0:
        return u.copy(), 0.0
    denom = float(np.dot(direction, direction))
    if not np.isfinite(denom) or denom <= 1e-12:
        raise ValueError("no usable kinematic correction direction")
    delta = (violation / denom) * direction
    projected = u + delta
    magnitude = float(np.linalg.norm(delta))
    if not np.all(np.isfinite(projected)) or not np.isfinite(magnitude):
        raise ValueError("kinematic correction is not finite")
    return projected, magnitude
