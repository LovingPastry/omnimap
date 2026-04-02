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
    """Create a thin box used as a simple ground plane.

    Args:
        size: Side length of the square ground in world units (meters).
        z: Top surface height of the ground plane in world frame.
        thickness: Thickness of the box. Must be > 0.
        color: RGB color in [0, 1].

    Returns:
        A painted Open3D triangle mesh whose top surface lies at ``z``.
    """
    if size <= 0:
        raise ValueError(f"size must be positive, got {size}")
    if thickness <= 0:
        raise ValueError(f"thickness must be positive, got {thickness}")

    mesh = o3d.geometry.TriangleMesh.create_box(width=size, height=size, depth=thickness)
    # Open3D box is created in +x,+y,+z from origin. Shift so the top face is at z
    # and the plane is centered around the world origin in x/y.
    mesh.translate(np.array([-size / 2.0, -size / 2.0, z - thickness], dtype=np.float64))
    mesh.paint_uniform_color(tuple(float(c) for c in color))
    mesh.compute_vertex_normals()
    return mesh


def create_coordinate_frame(size: float = 0.2) -> o3d.geometry.TriangleMesh:
    """Create a coordinate frame mesh for debugging camera/world directions."""
    if size <= 0:
        raise ValueError(f"size must be positive, got {size}")
    return o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)


def ensure_rgb_colors(colors: np.ndarray, default_color: Color3 = (0.7, 0.7, 0.7)) -> np.ndarray:
    """Ensure point colors are available and clipped to [0, 1].

    Args:
        colors: Array of shape [N, 3] or empty.
        default_color: RGB color used when colors are unavailable.

    Returns:
        Color array of shape [N, 3] in float64 and [0, 1].
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
