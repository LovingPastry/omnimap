"""Two-dimensional spherical-rate projections and frozen Servo models."""

import numpy as np


def project_feasible_spherical_command(u_info, a, c, *, linear_matrix=None,
                                       linear_bounds=None, norm_limits=(),
                                       local_center=None, local_radius=None):
    """Minimize ||u-u_info|| over one simultaneous convex feasible set.

    All matrices act on dimensionless rates u. Linear rows mean L u <= h
    (e.g. both signs of joint velocity bounds). Each norm limit is (M, b, r)
    and means ||M u+b|| <= r (camera speed or per-cycle velocity change).
    The optional local ball bounds the validity of the execution-side model.
    No post-solve clipping is permitted. Invalid, infeasible or unverified
    solutions raise ValueError; callers must not execute a partial result.
    """
    # Retain exact, dependency-free handling of the basic half-space.
    basic, _ = project_spherical_command(u_info, a, c)
    u = np.asarray(u_info, dtype=float)
    rows = [-np.asarray(a, dtype=float)]
    bounds = [-float(c)]
    if (linear_matrix is None) != (linear_bounds is None):
        raise ValueError("linear matrix and bounds must be supplied together")
    if linear_matrix is not None:
        matrix = np.asarray(linear_matrix, dtype=float)
        rhs = np.asarray(linear_bounds, dtype=float)
        if (matrix.ndim != 2 or matrix.shape[1] != 2 or rhs.shape != (len(matrix),)
                or not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(rhs))):
            raise ValueError("invalid linear feasible set")
        rows.extend(matrix)
        bounds.extend(rhs)
    balls = list(norm_limits)
    if (local_center is None) != (local_radius is None):
        raise ValueError("local center and radius must be supplied together")
    if local_center is not None:
        center = np.asarray(local_center, dtype=float)
        if center.shape != (2,):
            raise ValueError("local center must contain two rates")
        balls.append((np.eye(2), -center, local_radius))
    # Normalize constraints to keep physical units out of solver tolerances.
    linear = []
    for row, bound in zip(rows, bounds):
        scale = np.linalg.norm(row)
        if scale == 0:
            if bound < 0:
                raise ValueError("infeasible constant linear constraint")
            continue
        if not np.isfinite(scale):
            raise ValueError("invalid linear constraint scale")
        linear.append((row / scale, bound / scale))
    quadratic = []
    for matrix, bias, radius in balls:
        matrix, bias = np.asarray(matrix, dtype=float), np.asarray(bias, dtype=float)
        radius = float(radius)
        if (matrix.ndim != 2 or matrix.shape[1] != 2 or matrix.shape[0] == 0
                or bias.shape != (matrix.shape[0],) or not np.all(np.isfinite(matrix))
                or not np.all(np.isfinite(bias)) or not np.isfinite(radius) or radius <= 0):
            raise ValueError("norm constraints require finite matrices and positive radii")
        matrix, bias = matrix / radius, bias / radius
        if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(bias)):
            raise ValueError("invalid normalized norm constraint")
        quadratic.append((matrix, bias))

    def values(x):
        return np.asarray([bound - row @ x for row, bound in linear]
                          + [1.0 - np.dot(m @ x + b, m @ x + b) for m, b in quadratic])

    def jacobian(x):
        return np.asarray([-row for row, _ in linear]
                          + [-2.0 * m.T @ (m @ x + b) for m, b in quadratic]).reshape(-1, 2)

    # If the half-space optimum meets all other limits it is the joint optimum.
    if np.all(values(basic) >= 0):
        return basic, float(np.linalg.norm(basic-u))
    try:
        from scipy.optimize import minimize, nnls
    except ImportError as exc:
        raise ValueError("joint projection requires scipy") from exc
    result = minimize(lambda x: .5 * np.dot(x-u, x-u), basic,
                      jac=lambda x: x-u, method="SLSQP",
                      constraints=[dict(type="ineq", fun=values, jac=jacobian)],
                      options=dict(ftol=1e-10, maxiter=100))
    point = np.asarray(result.x)
    slack = values(point)
    if (not result.success or not np.all(np.isfinite(point))
            or not np.all(np.isfinite(slack)) or np.any(slack < -1e-9)):
        raise ValueError("joint projection infeasible or solver failed")
    # Independently check convex KKT stationarity and complementarity. A solver
    # success flag alone is not evidence of a minimum-distance feasible result.
    active = slack <= 1e-7
    gradient = point-u
    if np.any(active):
        try:
            multipliers, residual = nnls(jacobian(point)[active].T, gradient)
        except (ValueError, RuntimeError) as exc:
            raise ValueError("projection optimality check failed") from exc
        complementarity = np.max(np.abs(multipliers * slack[active]))
    else:
        residual, complementarity = np.linalg.norm(gradient), 0.0
    tolerance = 1e-7 * (1.0 + np.linalg.norm(gradient))
    if not np.isfinite(residual) or residual > tolerance or complementarity > tolerance:
        raise ValueError("projection optimum not verified")
    return point, float(np.linalg.norm(gradient))


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


def unclipped_preview_rate_model(rate_model, rates):
    """Affine candidate before linear limits, exclusively for non-actuating use.

    Speed/slew/deadband belong in the joint feasible set. A clipped source
    command is not an affine execution trace; its pre-limit mathematical map
    can still be queried, then projected and checked against actual output.
    """
    valid, reason, matrix, bias = rate_model
    allowed = {"linear_speed_limit_active", "linear_slew_limit_active", "linear_deadband_active"}
    matrix, bias, rates = np.asarray(matrix), np.asarray(bias), np.asarray(rates)
    if (not valid and reason not in allowed or matrix.shape != (6, 2)
            or bias.shape != (6,) or rates.shape != (2,)
            or not all(np.all(np.isfinite(x)) for x in (matrix, bias, rates))):
        return False, "preview_affine_model_unavailable:" + reason, matrix, bias
    return True, "unclipped_preview:" + reason, matrix.copy(), bias.copy()
