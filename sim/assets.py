"""仿真资源构造模块。

本模块提供场景搭建所需的基础几何体与颜色处理工具，包括地面平面、
坐标轴网格以及点云颜色规范化函数。其目标是集中管理可复用的资产构造
逻辑，减少上层场景模拟代码中的重复实现。
"""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
import open3d as o3d


Color3 = Tuple[float, float, float]


def create_ground_plane(
    size: float = 4.0,
    z: float = 0.0,
    thickness: float = 0.01,
    color: Color3 = (0.5, 0.5, 0.5),
) -> o3d.geometry.TriangleMesh:
    """创建一个薄盒体，作为简易地面平面。

    参数:
        size: 正方形地面边长，单位为世界坐标（米）。
        z: 地面顶面在世界坐标系下的高度。
        thickness: 盒体厚度，必须大于 0。
        color: RGB 颜色，取值范围 [0, 1]。

    返回:
        一个已着色的 Open3D 三角网格，其顶面位于 ``z``。
    """
    if size <= 0:
        raise ValueError(f"size must be positive, got {size}")
    if thickness <= 0:
        raise ValueError(f"thickness must be positive, got {thickness}")

    mesh = o3d.geometry.TriangleMesh.create_box(width=size, height=size, depth=thickness)
    # Open3D 创建的盒体默认从原点沿 +x/+y/+z 方向展开。
    # 这里将其平移到 x/y 围绕世界原点居中，且顶面高度对齐到 z。
    mesh.translate(np.array([-size / 2.0, -size / 2.0, z - thickness], dtype=np.float64))
    mesh.paint_uniform_color(tuple(float(c) for c in color))
    mesh.compute_vertex_normals()
    return mesh


def create_coordinate_frame(size: float = 0.2) -> o3d.geometry.TriangleMesh:
    """创建坐标轴网格，用于调试相机/世界坐标方向。"""
    if size <= 0:
        raise ValueError(f"size must be positive, got {size}")
    return o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)


def ensure_rgb_colors(colors: np.ndarray, default_color: Color3 = (0.7, 0.7, 0.7)) -> np.ndarray:
    """确保点云颜色可用，并裁剪到 [0, 1] 区间。

    参数:
        colors: 形状为 [N, 3] 的颜色数组，或空数组。
        default_color: 当颜色不可用时使用的默认 RGB 颜色。

    返回:
        形状为 [N, 3] 的颜色数组，类型为 float64，取值范围 [0, 1]。
    """
    colors = np.asarray(colors)
    if colors.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    if colors.ndim != 2 or colors.shape[1] != 3:
        raise ValueError(f"colors must have shape [N, 3], got {colors.shape}")
    colors = colors.astype(np.float64, copy=False)
    if not np.isfinite(colors).all():
        mask = np.isfinite(colors).all(axis=1)
        colors = colors[mask]
    if colors.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    if colors.max() > 1.0:
        colors = colors / 255.0
    return np.clip(colors, 0.0, 1.0)
