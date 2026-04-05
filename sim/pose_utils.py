"""Pose conversion helpers for the simulation -> OmniMap bridge.

This module centralizes all pose conversions used by Phase 2 so the
simulation side does not need to guess about coordinate conventions.

Conventions used here:
- `c2w`: camera-to-world 4x4 homogeneous transform.
- `w2c`: world-to-camera 4x4 homogeneous transform.
- `posevec`: 7D vector `[tx, ty, tz, qx, qy, qz, qw]` describing `w2c`.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from scipy.spatial.transform import Rotation as R


def _as_matrix44(matrix: np.ndarray, name: str) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"{name} must have shape (4, 4), got {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} contains NaN or Inf")
    return matrix


def _as_posevec(posevec: np.ndarray, name: str = "posevec") -> np.ndarray:
    posevec = np.asarray(posevec, dtype=np.float64).reshape(-1)
    if posevec.shape != (7,):
        raise ValueError(f"{name} must have shape (7,), got {posevec.shape}")
    if not np.isfinite(posevec).all():
        raise ValueError(f"{name} contains NaN or Inf")
    return posevec


def c2w_to_w2c(c2w: np.ndarray) -> np.ndarray:
    """Convert camera-to-world to world-to-camera."""
    return np.linalg.inv(_as_matrix44(c2w, "c2w"))


def w2c_to_c2w(w2c: np.ndarray) -> np.ndarray:
    """Convert world-to-camera to camera-to-world."""
    return np.linalg.inv(_as_matrix44(w2c, "w2c"))


def w2c_to_posevec(w2c: np.ndarray) -> np.ndarray:
    """Convert a `w2c` matrix into OmniMap's 7D pose vector."""
    w2c = _as_matrix44(w2c, "w2c")
    quat_xyzw = R.from_matrix(w2c[:3, :3]).as_quat()
    return np.concatenate([w2c[:3, 3], quat_xyzw], axis=0)


def posevec_to_se3(posevec: np.ndarray) -> np.ndarray:
    """Convert OmniMap's 7D pose vector into a `w2c` 4x4 matrix."""
    posevec = _as_posevec(posevec)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = R.from_quat(posevec[3:]).as_matrix()
    matrix[:3, 3] = posevec[:3]
    return matrix


def posevec_to_w2c(posevec: np.ndarray) -> np.ndarray:
    """Alias for `posevec_to_se3` for clearer call sites."""
    return posevec_to_se3(posevec)


def posevec_to_c2w(posevec: np.ndarray) -> np.ndarray:
    """Convert OmniMap's pose vector into a `c2w` matrix."""
    return w2c_to_c2w(posevec_to_w2c(posevec))


def c2w_to_posevec(c2w: np.ndarray) -> np.ndarray:
    """Convert a `c2w` matrix into OmniMap's 7D `w2c` pose vector."""
    return w2c_to_posevec(c2w_to_w2c(c2w))


def validate_pose_roundtrip(c2w: np.ndarray, atol: float = 1e-6) -> Dict[str, float]:
    """Validate the required Phase-2 conversion chain.

    Validation chain:
    `c2w -> w2c -> posevec -> SE3(matrix) -> c2w`
    """
    c2w = _as_matrix44(c2w, "c2w")
    w2c = c2w_to_w2c(c2w)
    posevec = w2c_to_posevec(w2c)
    w2c_recovered = posevec_to_se3(posevec)
    c2w_recovered = w2c_to_c2w(w2c_recovered)

    w2c_error = float(np.max(np.abs(w2c - w2c_recovered)))
    c2w_error = float(np.max(np.abs(c2w - c2w_recovered)))
    valid = bool(w2c_error <= atol and c2w_error <= atol)
    return {
        "valid": valid,
        "w2c_max_abs_err": w2c_error,
        "c2w_max_abs_err": c2w_error,
        "atol": float(atol),
    }


def assert_pose_roundtrip(c2w: np.ndarray, atol: float = 1e-6) -> Dict[str, float]:
    """Raise if the Phase-2 pose roundtrip deviates beyond tolerance."""
    stats = validate_pose_roundtrip(c2w, atol=atol)
    if not bool(stats["valid"]):
        raise ValueError(
            "Pose conversion roundtrip failed: "
            f"w2c_err={stats['w2c_max_abs_err']:.3e}, "
            f"c2w_err={stats['c2w_max_abs_err']:.3e}, atol={stats['atol']:.3e}"
        )
    return stats
