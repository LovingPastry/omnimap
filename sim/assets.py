"""仿真资源构造模块。

本模块只保留当前仿真实验真正会复用的基础几何体构造逻辑，
避免把地面和坐标轴的细节散落到多个入口脚本中。
"""

from __future__ import annotations

from typing import Tuple
import open3d as o3d
import numpy as np

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

    mesh = o3d.geometry.TriangleMesh.create_box(
        width=size, height=size, depth=thickness
    )
    # Open3D 创建的盒体默认从原点沿 +x/+y/+z 方向展开。
    # 这里将其平移到 x/y 围绕世界原点居中，且顶面高度对齐到 z。
    mesh.translate(
        np.array([-size / 2.0, -size / 2.0, z - thickness], dtype=np.float64)
    )
    mesh.paint_uniform_color(tuple(float(c) for c in color))
    mesh.compute_vertex_normals()
    return mesh


def create_coordinate_frame(size: float = 0.2) -> o3d.geometry.TriangleMesh:
    """创建坐标轴网格，用于调试相机/世界坐标方向。"""
    if size <= 0:
        raise ValueError(f"size must be positive, got {size}")
    return o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
