from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import open3d as o3d

from sim.assets import create_coordinate_frame, create_ground_plane


@dataclass
class RenderResult:
    """RGBD render output from the simulator.

    Attributes:
        rgb: RGB image with shape [H, W, 3], dtype uint8.
        depth: Depth image with shape [H, W], dtype float32, unit meters.
    """

    rgb: np.ndarray
    depth: np.ndarray


class SceneSimulator:
    """Open3D-based static scene simulator for RGBD rendering.

    This class is the Phase-1 input source of the closed-loop simulation:
    it manages a static scene (point cloud + optional ground + debug geometries)
    and renders RGBD images from a given camera pose.

    Coordinate convention used by the public API:
    - input pose is ``c2w`` (camera-to-world), shape [4, 4]
    - internal Open3D renderer camera setup uses ``w2c`` (world-to-camera)

    The conversion is handled internally. Callers should always provide ``c2w``.
    """

    def __init__(self, width: int = 640, height: int = 480, background=(0.0, 0.0, 0.0, 1.0)):
        if width <= 0 or height <= 0:
            raise ValueError(f"width/height must be positive, got width={width}, height={height}")

        self.width = int(width)
        self.height = int(height)
        self.background = np.asarray(background, dtype=np.float32)
        if self.background.shape != (4,):
            raise ValueError(f"background must have shape (4,), got {self.background.shape}")

        self.renderer = o3d.visualization.rendering.OffscreenRenderer(self.width, self.height)
        self.renderer.scene.set_background(self.background)
        self.renderer.scene.scene.enable_sun_light(False)
        self.renderer.scene.scene.enable_indirect_light(True)

        self.fx: Optional[float] = None
        self.fy: Optional[float] = None
        self.cx: Optional[float] = None
        self.cy: Optional[float] = None

        self.pointcloud: Optional[o3d.geometry.PointCloud] = None
        self.pointcloud_path: Optional[Path] = None
        self.ground: Optional[o3d.geometry.TriangleMesh] = None
        self.coord_frame: Optional[o3d.geometry.TriangleMesh] = None

        self.scene_center: Optional[np.ndarray] = None
        self.aabb: Optional[o3d.geometry.AxisAlignedBoundingBox] = None
        self._geometry_names = set()

        self._default_material = o3d.visualization.rendering.MaterialRecord()
        self._default_material.shader = "defaultUnlit"
        self._ground_material = o3d.visualization.rendering.MaterialRecord()
        self._ground_material.shader = "defaultLit"

    # ---------------------------------------------------------------------
    # Geometry management
    # ---------------------------------------------------------------------
    def clear_scene(self) -> None:
        """Remove all geometries from the renderer scene."""
        for name in list(self._geometry_names):
            self.renderer.scene.remove_geometry(name)
        self._geometry_names.clear()
        self.pointcloud = None
        self.pointcloud_path = None
        self.ground = None
        self.coord_frame = None
        self.scene_center = None
        self.aabb = None

    def _add_geometry(self, name: str, geometry, material) -> None:
        if name in self._geometry_names:
            self.renderer.scene.remove_geometry(name)
        self.renderer.scene.add_geometry(name, geometry, material)
        self._geometry_names.add(name)

    def load_pointcloud(self, path: str, voxel_size: Optional[float] = None) -> o3d.geometry.PointCloud:
        """Load a point cloud and add it to the rendering scene.

        Args:
            path: Path to a `.ply` or `.pcd` point cloud.
            voxel_size: Optional downsample voxel size in world units.

        Returns:
            The cleaned Open3D point cloud instance stored in the simulator.
        """
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Point cloud file does not exist: {path}")
        if path_obj.suffix.lower() not in {".ply", ".pcd"}:
            raise ValueError(f"Unsupported point cloud format: {path_obj.suffix}")

        pcd = o3d.io.read_point_cloud(str(path_obj))
        if pcd.is_empty():
            raise ValueError(f"Loaded point cloud is empty: {path}")

        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        finite_mask = np.isfinite(points).all(axis=1)
        if colors.size > 0:
            finite_mask &= np.isfinite(colors).all(axis=1)
        points = points[finite_mask]
        if colors.size > 0:
            colors = colors[finite_mask]

        if points.shape[0] == 0:
            raise ValueError(f"No valid finite points remain after cleaning: {path}")

        clean_pcd = o3d.geometry.PointCloud()
        clean_pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        if colors.size == 0:
            default_color = np.tile(np.array([[0.7, 0.7, 0.7]], dtype=np.float64), (points.shape[0], 1))
            clean_pcd.colors = o3d.utility.Vector3dVector(default_color)
        else:
            colors = colors.astype(np.float64, copy=False)
            if colors.max() > 1.0:
                colors = colors / 255.0
            colors = np.clip(colors, 0.0, 1.0)
            clean_pcd.colors = o3d.utility.Vector3dVector(colors)

        if voxel_size is not None:
            if voxel_size <= 0:
                raise ValueError(f"voxel_size must be positive, got {voxel_size}")
            clean_pcd = clean_pcd.voxel_down_sample(voxel_size=float(voxel_size))
            if clean_pcd.is_empty():
                raise ValueError("Point cloud became empty after voxel downsample")

        self.pointcloud = clean_pcd
        self.pointcloud_path = path_obj
        self.aabb = clean_pcd.get_axis_aligned_bounding_box()
        self.scene_center = np.asarray(self.aabb.get_center(), dtype=np.float64)

        self._add_geometry("scene_pointcloud", clean_pcd, self._default_material)
        return clean_pcd

    def add_ground(self, size: float = 4.0, z: float = 0.0, color=(0.5, 0.5, 0.5), thickness: float = 0.01):
        """Add a simple ground plane into the scene.

        Args:
            size: Side length of the square ground plane.
            z: World z value of the *top* face of the ground.
            color: Ground RGB color in [0, 1].
            thickness: Thickness of the supporting thin box.
        """
        self.ground = create_ground_plane(size=size, z=z, color=color, thickness=thickness)
        self._add_geometry("ground_plane", self.ground, self._ground_material)
        return self.ground

    def add_coordinate_frame(self, size: float = 0.2):
        """Add a coordinate frame for debugging scene/world directions."""
        self.coord_frame = create_coordinate_frame(size=size)
        self._add_geometry("coord_frame", self.coord_frame, self._ground_material)
        return self.coord_frame

    # ---------------------------------------------------------------------
    # Camera and rendering
    # ---------------------------------------------------------------------
    def set_intrinsics(self, fx: float, fy: float, cx: float, cy: float) -> None:
        """Set pinhole camera intrinsics used by the renderer."""
        for name, value in {"fx": fx, "fy": fy, "cx": cx, "cy": cy}.items():
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite, got {value}")
        if fx <= 0 or fy <= 0:
            raise ValueError(f"fx/fy must be positive, got fx={fx}, fy={fy}")
        self.fx = float(fx)
        self.fy = float(fy)
        self.cx = float(cx)
        self.cy = float(cy)

    def _assert_ready_to_render(self) -> None:
        if self.fx is None or self.fy is None or self.cx is None or self.cy is None:
            raise RuntimeError("Camera intrinsics are not set. Call set_intrinsics(...) first.")
        if self.pointcloud is None and self.ground is None:
            raise RuntimeError("Scene is empty. Load a point cloud or add geometry before rendering.")

    @staticmethod
    def _ensure_c2w(c2w: np.ndarray) -> np.ndarray:
        c2w = np.asarray(c2w, dtype=np.float64)
        if c2w.shape != (4, 4):
            raise ValueError(f"c2w must have shape (4, 4), got {c2w.shape}")
        if not np.isfinite(c2w).all():
            raise ValueError("c2w contains NaN or Inf")
        return c2w

    def render(self, c2w: np.ndarray) -> RenderResult:
        """Render RGBD from a camera pose.

        Args:
            c2w: Camera-to-world transform, shape [4, 4].

        Returns:
            RenderResult with:
            - rgb: uint8 RGB image of shape [H, W, 3]
            - depth: float32 depth image in meters of shape [H, W]

        Notes:
            Open3D OffscreenRenderer expects the extrinsic matrix in world-to-camera
            convention. This method therefore converts the caller-provided ``c2w``
            into ``w2c`` internally.
        """
        self._assert_ready_to_render()
        c2w = self._ensure_c2w(c2w)
        w2c = np.linalg.inv(c2w)

        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            self.width,
            self.height,
            self.fx,
            self.fy,
            self.cx,
            self.cy,
        )
        self.renderer.setup_camera(intrinsic, w2c)

        color = np.asarray(self.renderer.render_to_image())
        depth = np.asarray(self.renderer.render_to_depth_image(z_in_view_space=True))

        if color.ndim != 3 or color.shape[2] not in (3, 4):
            raise RuntimeError(f"Unexpected rendered color shape: {color.shape}")
        if color.shape[2] == 4:
            color = color[:, :, :3]
        if depth.ndim != 2:
            raise RuntimeError(f"Unexpected rendered depth shape: {depth.shape}")

        color = np.ascontiguousarray(color.astype(np.uint8))
        depth = np.ascontiguousarray(depth.astype(np.float32))
        return RenderResult(rgb=color, depth=depth)

    # ---------------------------------------------------------------------
    # Debug helpers
    # ---------------------------------------------------------------------
    def get_scene_stats(self) -> Dict[str, object]:
        """Return lightweight scene statistics for logging/debugging."""
        stats: Dict[str, object] = {
            "width": self.width,
            "height": self.height,
            "intrinsics": None if self.fx is None else [self.fx, self.fy, self.cx, self.cy],
            "has_pointcloud": self.pointcloud is not None,
            "has_ground": self.ground is not None,
        }
        if self.pointcloud is not None:
            stats["num_points"] = int(np.asarray(self.pointcloud.points).shape[0])
        if self.scene_center is not None:
            stats["scene_center"] = self.scene_center.tolist()
        if self.aabb is not None:
            stats["aabb_min"] = self.aabb.get_min_bound().tolist()
            stats["aabb_max"] = self.aabb.get_max_bound().tolist()
        return stats

    @staticmethod
    def depth_to_vis(depth: np.ndarray, min_depth: Optional[float] = None, max_depth: Optional[float] = None) -> np.ndarray:
        """Convert metric depth to a uint8 visualization image."""
        depth = np.asarray(depth, dtype=np.float32)
        if depth.ndim != 2:
            raise ValueError(f"depth must have shape [H, W], got {depth.shape}")

        valid = np.isfinite(depth) & (depth > 0)
        if not np.any(valid):
            return np.zeros((*depth.shape, 3), dtype=np.uint8)

        d_min = float(depth[valid].min()) if min_depth is None else float(min_depth)
        d_max = float(depth[valid].max()) if max_depth is None else float(max_depth)
        if d_max <= d_min:
            d_max = d_min + 1e-6

        norm = np.clip((depth - d_min) / (d_max - d_min), 0.0, 1.0)
        gray = (norm * 255.0).astype(np.uint8)
        return np.repeat(gray[:, :, None], 3, axis=2)
