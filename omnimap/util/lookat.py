"""Single source of truth for the hemisphere look-at convention.

Four sites used to build this pose independently and disagreed three ways: the
cross-product order differed (a constant 180 degree roll) and the world up axis
differed (a theta-dependent 0-180 degree roll). Since the Fisher score is scored
on one convention while the arm is commanded on another, that disagreement meant
the planner scored a differently-rolled camera than the servo achieved.

The convention here is ``up = [-1, 0, 0]``, matching what the servo already did.
Its virtue is behaviour at the pole: along a great circle through the north pole
this convention's roll is continuous (~0.2 degrees across the transit), whereas
``up = [0, 0, 1]`` flips by exactly 180 degrees. Because the servo rebuilds the
orientation locally every tick, that flip would become a real commanded roll of
pi radians on the arm. Its singularities sit at ``phi = 0, theta in {0, pi}``,
which the policy's ``phi_min`` never reaches, instead of at the pole, which is
precisely the region this control loop needs to open up.

This module deliberately imports nothing beyond numpy so both the ROS servo and
the ROS-free planner can share it.
"""

from __future__ import annotations

import numpy as np

# Camera y axis stores "down" in this project, so the c2w second column is -up.
DEFAULT_WORLD_UP = np.array([-1.0, 0.0, 0.0], dtype=np.float64)

# Tried in order when ``up`` is parallel to ``forward``. A fixed single fallback is
# not enough: it can coincide with the caller's ``up`` and stay degenerate (passing
# up=[0,0,1] at the pole hit exactly that). At least one of three orthogonal axes
# is always non-parallel to ``forward``.
_FALLBACK_WORLD_UPS = (
    np.array([0.0, 0.0, 1.0], dtype=np.float64),
    np.array([0.0, 1.0, 0.0], dtype=np.float64),
    np.array([1.0, 0.0, 0.0], dtype=np.float64),
)


def look_at_c2w(
    eye: np.ndarray,
    target: np.ndarray,
    up: np.ndarray | None = None,
) -> np.ndarray:
    """Build a ``c2w`` whose optical (+z) axis points from ``eye`` to ``target``.

    Columns are ``[right, down, forward, eye]``.
    """
    eye = np.asarray(eye, dtype=np.float64).reshape(3)
    target = np.asarray(target, dtype=np.float64).reshape(3)
    up = DEFAULT_WORLD_UP if up is None else np.asarray(up, dtype=np.float64).reshape(3)

    forward = target - eye
    forward_norm = float(np.linalg.norm(forward))
    if forward_norm < 1e-12:
        raise ValueError("eye and target are too close; cannot build look-at pose")
    forward = forward / forward_norm

    # right = forward x up. All four legacy sites are being normalized onto this
    # order; camera_utils used the reverse, which is where the constant 180
    # degree roll between "what Fisher scored" and "what moved" came from.
    right = np.cross(forward, up)
    right_norm = float(np.linalg.norm(right))
    if right_norm < 1e-12:
        for fallback_up in _FALLBACK_WORLD_UPS:
            right = np.cross(forward, fallback_up)
            right_norm = float(np.linalg.norm(right))
            if right_norm >= 1e-12:
                break
        else:
            raise ValueError("failed to construct a valid right axis for look-at pose")
    right = right / right_norm

    true_up = np.cross(right, forward)
    true_up = true_up / max(float(np.linalg.norm(true_up)), 1e-12)

    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, 0] = right
    c2w[:3, 1] = -true_up
    c2w[:3, 2] = forward
    c2w[:3, 3] = eye
    return c2w
