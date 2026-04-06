"""仿真到 OmniMap 桥接的位姿转换工具。

本模块集中管理 Phase 2 使用的所有位姿转换，
避免仿真侧反复猜测坐标系约定。

本模块约定：
- `c2w`：camera-to-world 的 4x4 齐次变换。
- `w2c`：world-to-camera 的 4x4 齐次变换。
- `posevec`：描述 `w2c` 的 7 维向量 `[tx, ty, tz, qx, qy, qz, qw]`。
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
    """将 camera-to-world 转为 world-to-camera。"""
    return np.linalg.inv(_as_matrix44(c2w, "c2w"))


def w2c_to_c2w(w2c: np.ndarray) -> np.ndarray:
    """将 world-to-camera 转为 camera-to-world。"""
    return np.linalg.inv(_as_matrix44(w2c, "w2c"))


def w2c_to_posevec(w2c: np.ndarray) -> np.ndarray:
    """将 `w2c` 矩阵转换为 OmniMap 的 7 维位姿向量。"""
    w2c = _as_matrix44(w2c, "w2c")
    quat_xyzw = R.from_matrix(w2c[:3, :3]).as_quat()
    return np.concatenate([w2c[:3, 3], quat_xyzw], axis=0)


def posevec_to_se3(posevec: np.ndarray) -> np.ndarray:
    """将 OmniMap 的 7 维位姿向量转换为 `w2c` 4x4 矩阵。"""
    posevec = _as_posevec(posevec)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = R.from_quat(posevec[3:]).as_matrix()
    matrix[:3, 3] = posevec[:3]
    return matrix


def posevec_to_w2c(posevec: np.ndarray) -> np.ndarray:
    """`posevec_to_se3` 的别名，用于提升调用处可读性。"""
    return posevec_to_se3(posevec)


def posevec_to_c2w(posevec: np.ndarray) -> np.ndarray:
    """将 OmniMap 位姿向量转换为 `c2w` 矩阵。"""
    return w2c_to_c2w(posevec_to_w2c(posevec))


def c2w_to_posevec(c2w: np.ndarray) -> np.ndarray:
    """将 `c2w` 矩阵转换为 OmniMap 的 7 维 `w2c` 位姿向量。"""
    return w2c_to_posevec(c2w_to_w2c(c2w))


def validate_pose_roundtrip(c2w: np.ndarray, atol: float = 1e-6) -> Dict[str, float]:
    """验证 Phase 2 所需的转换链路。

    验证链路：
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
    """当 Phase 2 位姿往返误差超过容差时抛出异常。"""
    stats = validate_pose_roundtrip(c2w, atol=atol)
    if not bool(stats["valid"]):
        raise ValueError(
            "Pose conversion roundtrip failed: "
            f"w2c_err={stats['w2c_max_abs_err']:.3e}, "
            f"c2w_err={stats['c2w_max_abs_err']:.3e}, atol={stats['atol']:.3e}"
        )
    return stats
