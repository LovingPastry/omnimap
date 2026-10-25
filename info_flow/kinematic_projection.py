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


def camera_rate_model(*, radius, e_theta, e_phi, rates, radial_velocity,
                      linear_command, angular_command, previous_linear,
                      linear_speed_limit, linear_step_limit, linear_deadband):
    """Frozen Servo state: twist_camera = B @ physical_rates + bias.

    Angular correction and radial PI are independent of the two requested
    rates at the current pose. Reject nonlinear linear-velocity branches and
    their boundaries; this local model is not a guarantee for a finite change.
    Zero/nonpositive speed and slew limits have the runtime's disabled meaning.
    """
    b = np.r_[np.asarray(radial_velocity, dtype=float),
              np.asarray(angular_command, dtype=float)]
    matrix = np.zeros((6, 2), dtype=float)
    matrix[:3, 0] = float(radius) * np.asarray(e_theta, dtype=float)
    matrix[:3, 1] = float(radius) * np.asarray(e_phi, dtype=float)
    rates = np.asarray(rates, dtype=float)
    actual = np.asarray(linear_command, dtype=float)
    previous = np.asarray(previous_linear, dtype=float)
    scalars = [radius, linear_speed_limit, linear_step_limit, linear_deadband]
    if (b.shape != (6,) or rates.shape != (2,) or actual.shape != (3,) or previous.shape != (3,)
            or not all(np.all(np.isfinite(x)) for x in (matrix, b, rates, actual, previous, scalars))
            or radius <= 1e-12):
        return False, "rate_model_nonfinite_or_degenerate", matrix, b
    raw = (matrix @ rates + b)[:3]
    norm, step = np.linalg.norm(raw), np.linalg.norm(raw - previous)
    if linear_speed_limit > 0 and norm >= linear_speed_limit - 1e-12:
        return False, "linear_speed_limit_active", matrix, b
    if linear_step_limit > 0 and step >= linear_step_limit - 1e-12:
        return False, "linear_slew_limit_active", matrix, b
    if linear_deadband > 0 and norm <= linear_deadband + 1e-12:
        return False, "linear_deadband_active", matrix, b
    if not np.allclose(raw, actual, rtol=1e-10, atol=1e-12):
        return False, "rate_model_output_mismatch", matrix, b
    return True, "ok", matrix, b
