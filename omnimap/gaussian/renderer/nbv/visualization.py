from __future__ import annotations
import os
from typing import Optional

import numpy as np
import open3d as o3d

from util.utils import Log
from util.vis_utils import draw_camera
from gaussian.utils.sh_utils import SH2RGB


class FisherVisualizer:
    def __init__(self, gaussians, config, save_dir: str, vis_gui: bool):
        self.gaussians = gaussians
        self.config = config
        self.save_dir = save_dir
        self.vis_gui = vis_gui
        self.heatmap_window = None
        self.velocity_window = None
        self.heatmap_geometry = None
        self.fisher_hemi_geometry = None
        self.velocity_surface_geometry = None
        self._heatmap_window_initialized = False
        self._velocity_window_initialized = False
        self.fisher_frame0_exported = False
        self.last_gs_points = None
        self.last_gs_colors = None
        self.last_fisher_points = None
        self.last_fisher_colors = None
        self.last_tsdf_points = None
        self.last_tsdf_colors = None
        self.last_camera_pose = None
        self.last_traj_points = None
        self.last_velocity_points = None
        self.last_velocity_colors = None

        self.velocity_geometry = None
        self.heatmap_context_geometries = []
        self.velocity_context_geometries = []

    @staticmethod
    def _build_arrow_mesh(
        start: np.ndarray,
        direction: np.ndarray,
        arrow_len: float,
        color: Optional[np.ndarray] = None,
    ):
        direction = np.asarray(direction, dtype=np.float64)
        norm = np.linalg.norm(direction)
        if norm < 1e-12:
            return None

        unit_dir = direction / norm
        cone_height = arrow_len * 0.35
        cylinder_height = max(arrow_len - cone_height, 1e-6)
        arrow = o3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius=arrow_len * 0.04,
            cone_radius=arrow_len * 0.08,
            cylinder_height=cylinder_height,
            cone_height=cone_height,
        )
        arrow_color = (
            np.asarray(color, dtype=np.float64).reshape(3)
            if color is not None
            else np.array([1.0, 0.0, 0.0], dtype=np.float64)
        )
        arrow.paint_uniform_color(arrow_color.tolist())

        z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        cross = np.cross(z_axis, unit_dir)
        cross_norm = np.linalg.norm(cross)
        dot = float(np.clip(np.dot(z_axis, unit_dir), -1.0, 1.0))

        if cross_norm > 1e-12:
            axis = cross / cross_norm
            angle = np.arccos(dot)
            rot = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle)
            arrow.rotate(rot, center=np.zeros(3, dtype=np.float64))
        elif dot < 0.0:
            rot = o3d.geometry.get_rotation_matrix_from_axis_angle(
                np.array([1.0, 0.0, 0.0], dtype=np.float64) * np.pi
            )
            arrow.rotate(rot, center=np.zeros(3, dtype=np.float64))

        arrow.translate(np.asarray(start, dtype=np.float64))
        return arrow

    def attach_windows(self, heatmap_window=None, velocity_window=None):
        self.heatmap_window = heatmap_window
        self.velocity_window = velocity_window
        self._heatmap_window_initialized = False
        self._velocity_window_initialized = False

    @staticmethod
    def _refresh_window(window):
        if window is not None:
            window.poll_events()
            window.update_renderer()

    def update_gaussian_cache(self, gs_points=None, gs_colors=None):
        if gs_points is not None and gs_colors is not None:
            self.last_gs_points = np.asarray(gs_points).copy()
            self.last_gs_colors = np.asarray(gs_colors).copy()
            return

        opacity = self.gaussians.get_opacity.detach().squeeze().cpu().numpy()
        mask = opacity > 0.3
        gs_points = self.gaussians.get_xyz.detach().cpu().numpy()[mask]
        gs_colors = (
            SH2RGB(self.gaussians.get_features.detach()).squeeze().cpu().numpy()[mask]
        )
        if self.config.get("scene") == "room_0":
            z_mask = gs_points[:, 2] <= 0.7
            gs_points = gs_points[z_mask]
            gs_colors = gs_colors[z_mask]
        self.last_gs_points = gs_points
        self.last_gs_colors = gs_colors

    def update_context(
        self,
        *,
        tsdf_points=None,
        tsdf_colors=None,
        camera_pose=None,
        traj_points=None,
    ) -> None:
        """Cache shared scene context used by split Fisher windows."""
        if tsdf_points is not None and tsdf_colors is not None:
            self.last_tsdf_points = np.asarray(tsdf_points).copy()
            self.last_tsdf_colors = np.asarray(tsdf_colors).copy()
        if camera_pose is not None:
            self.last_camera_pose = np.asarray(camera_pose, dtype=np.float64).copy()
        if traj_points is not None:
            self.last_traj_points = np.asarray(traj_points, dtype=np.float64).copy()

    def _clear_context_geometries(self, window, geometries):
        if window is None:
            geometries.clear()
            return
        for geometry in geometries:
            window.remove_geometry(geometry, reset_bounding_box=False)
        geometries.clear()

    def _build_context_geometries(self):
        geometries = []
        if self.last_tsdf_points is not None and len(self.last_tsdf_points) > 0:
            tsdf_pc = o3d.geometry.PointCloud()
            tsdf_pc.points = o3d.utility.Vector3dVector(self.last_tsdf_points)
            tsdf_pc.colors = o3d.utility.Vector3dVector(self.last_tsdf_colors)
            geometries.append(tsdf_pc)
        if self.last_traj_points is not None and len(self.last_traj_points) > 0:
            traj = o3d.geometry.LineSet()
            traj.points = o3d.utility.Vector3dVector(self.last_traj_points)
            if len(self.last_traj_points) >= 2:
                lines = [[i, i + 1] for i in range(len(self.last_traj_points) - 1)]
                traj.lines = o3d.utility.Vector2iVector(lines)
                traj.colors = o3d.utility.Vector3dVector(
                    np.tile(np.array([[1.0, 0.0, 0.0]], dtype=np.float64), (len(lines), 1))
                )
            geometries.append(traj)
        if self.last_camera_pose is not None:
            cam_lines, cam_plane = draw_camera(self.last_camera_pose)
            geometries.extend([cam_lines, cam_plane])
        return geometries

    def _update_split_context_window(self, window, geometry_store):
        window_initialized = len(geometry_store) > 0
        self._clear_context_geometries(window, geometry_store)
        if window is None:
            return
        for geometry_index, geometry in enumerate(self._build_context_geometries()):
            window.add_geometry(
                geometry,
                reset_bounding_box=(geometry_index == 0 and not window_initialized),
            )
            geometry_store.append(geometry)

    def apply_field_result(self, field_result):
        center = field_result.base_hemi.center.detach().float()
        radius = float(field_result.base_hemi.radius)
        display_radius_scale = float(
            self.config.get("fisher_display_radius_scale", 0.92)
        )
        arrow_radius_scale = float(
            self.config.get("fisher_arrow_radius_scale", display_radius_scale)
        )
        dense_points = (
            center[None, :] + (radius * display_radius_scale) * field_result.dense_dirs
        )

        show_fisher_heatmap = bool(self.config.get("show_fisher_heatmap", True))
        show_velocity_field = bool(self.config.get("show_velocity_field", False))
        fisher_window_mode = str(self.config.get("fisher_window_mode", "combined"))

        hemi_pc = o3d.geometry.PointCloud()
        hemi_pc.points = o3d.utility.Vector3dVector(dense_points.detach().cpu().numpy())
        hemi_pc.colors = o3d.utility.Vector3dVector(
            field_result.dense_colors.cpu().numpy()
        )

        # --- Velocity field visualization (optional) ---
        vel_mesh = None
        vel_pc = None
        if show_velocity_field:
            vel_dirs_tensor = getattr(field_result, "dense_vel_dirs", None)
            vel_points_tensor = getattr(field_result, "dense_dirs", None)
            vel_colors_tensor = getattr(field_result, "dense_velocity_colors", None)
            if vel_dirs_tensor is None or vel_points_tensor is None:
                vel_dirs_tensor = getattr(field_result, "sample_vel_dirs", None)
                vel_points_tensor = getattr(field_result, "sample_dirs", None)

            if vel_dirs_tensor is not None and vel_points_tensor is not None:
                vel_points = (
                    center[None, :]
                    + (radius * arrow_radius_scale) * vel_points_tensor
                ).detach().cpu().numpy()
                vel_dirs = vel_dirs_tensor.detach().cpu().numpy()
                vel_colors = (
                    vel_colors_tensor.detach().cpu().numpy()
                    if vel_colors_tensor is not None
                    else np.tile(
                        np.array([[1.0, 0.0, 0.0]], dtype=np.float64),
                        (len(vel_points), 1),
                    )
                )

                # Arrow length stays fixed; color now carries gradient magnitude.
                arrow_len = float(self.config.get("velocity_arrow_length", 0.07))
                # If user does not specify, interpret as relative to hemisphere radius when <=1.
                if arrow_len <= 1.0:
                    arrow_len = arrow_len * radius

                vel_pc = o3d.geometry.PointCloud()
                vel_pc.points = o3d.utility.Vector3dVector(vel_points)
                vel_pc.colors = o3d.utility.Vector3dVector(vel_colors)

                arrow_meshes = []
                for start, direction, color in zip(vel_points, vel_dirs, vel_colors):
                    arrow = self._build_arrow_mesh(start, direction, arrow_len, color=color)
                    if arrow is not None:
                        arrow_meshes.append(arrow)

                if len(arrow_meshes) > 0:
                    vel_mesh = arrow_meshes[0]
                    for arrow in arrow_meshes[1:]:
                        vel_mesh += arrow
                    vel_mesh.compute_vertex_normals()

        if self.vis_gui:
            if fisher_window_mode == "split":
                self._update_split_context_window(
                    self.heatmap_window, self.heatmap_context_geometries
                )
                self._update_split_context_window(
                    self.velocity_window, self.velocity_context_geometries
                )
            if self.heatmap_window is not None and self.heatmap_geometry is not None:
                self.heatmap_window.remove_geometry(
                    self.heatmap_geometry, reset_bounding_box=False
                )
            if show_fisher_heatmap and self.heatmap_window is not None:
                self.heatmap_geometry = hemi_pc
                self.fisher_hemi_geometry = hemi_pc
                self.heatmap_window.add_geometry(
                    self.heatmap_geometry,
                    reset_bounding_box=not self._heatmap_window_initialized,
                )
                self._heatmap_window_initialized = True
            else:
                self.heatmap_geometry = None
                self.fisher_hemi_geometry = None

            if self.velocity_window is not None and self.velocity_geometry is not None:
                self.velocity_window.remove_geometry(
                    self.velocity_geometry, reset_bounding_box=False
                )
                self.velocity_geometry = None
            if (
                self.velocity_window is not None
                and self.velocity_surface_geometry is not None
            ):
                self.velocity_window.remove_geometry(
                    self.velocity_surface_geometry, reset_bounding_box=False
                )
                self.velocity_surface_geometry = None

            if show_velocity_field and vel_mesh is not None and self.velocity_window is not None:
                show_velocity_surface = (
                    fisher_window_mode == "split"
                    or not show_fisher_heatmap
                    or self.velocity_window is not self.heatmap_window
                )
                if show_velocity_surface and vel_pc is not None:
                    self.velocity_surface_geometry = vel_pc
                    self.velocity_window.add_geometry(
                        self.velocity_surface_geometry,
                        reset_bounding_box=not self._velocity_window_initialized,
                    )
                self.velocity_geometry = vel_mesh
                self.velocity_window.add_geometry(
                    self.velocity_geometry,
                    reset_bounding_box=(
                        not self._velocity_window_initialized
                        and self.velocity_surface_geometry is None
                    ),
                )
                self._velocity_window_initialized = True

            refreshed = set()
            for window in (self.heatmap_window, self.velocity_window):
                if window is not None and id(window) not in refreshed:
                    self._refresh_window(window)
                    refreshed.add(id(window))

        self.last_fisher_points = np.asarray(hemi_pc.points).copy()
        self.last_fisher_colors = np.asarray(hemi_pc.colors).copy()
        self.last_velocity_points = (
            np.asarray(vel_pc.points).copy() if vel_pc is not None else None
        )
        self.last_velocity_colors = (
            np.asarray(vel_pc.colors).copy() if vel_pc is not None else None
        )
        if self.last_gs_points is None or self.last_gs_colors is None:
            self.update_gaussian_cache()
        return hemi_pc

    def _context_points_and_colors(self):
        if self.last_tsdf_points is not None and len(self.last_tsdf_points) > 0:
            return self.last_tsdf_points, self.last_tsdf_colors
        if self.last_gs_points is not None and len(self.last_gs_points) > 0:
            return self.last_gs_points, self.last_gs_colors
        return None, None

    @staticmethod
    def _write_point_cloud(path: str, points: np.ndarray, colors: np.ndarray) -> None:
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
        pc.colors = o3d.utility.Vector3dVector(np.asarray(colors, dtype=np.float64))
        o3d.io.write_point_cloud(path, pc)

    def export_current_artifacts(self, tag: str = "final") -> None:
        """Export the currently displayed Fisher heatmap/velocity views and backing geometry."""
        out_dir = os.path.join(self.save_dir, "nbv_vis")
        os.makedirs(out_dir, exist_ok=True)

        context_points, context_colors = self._context_points_and_colors()

        if context_points is not None and self.last_fisher_points is not None:
            merged_points = np.vstack([context_points, self.last_fisher_points])
            merged_colors = np.vstack([context_colors, self.last_fisher_colors])
            self._write_point_cloud(
                os.path.join(out_dir, f"{tag}_mapping_plus_fisher_heatmap.ply"),
                merged_points,
                merged_colors,
            )

        if context_points is not None and self.last_velocity_points is not None:
            merged_points = np.vstack([context_points, self.last_velocity_points])
            merged_colors = np.vstack([context_colors, self.last_velocity_colors])
            self._write_point_cloud(
                os.path.join(out_dir, f"{tag}_mapping_plus_velocity_surface.ply"),
                merged_points,
                merged_colors,
            )

        if self.velocity_geometry is not None:
            o3d.io.write_triangle_mesh(
                os.path.join(out_dir, f"{tag}_velocity_arrows.ply"),
                self.velocity_geometry,
            )

        if self.last_camera_pose is not None:
            np.save(
                os.path.join(out_dir, f"{tag}_camera_c2w.npy"),
                self.last_camera_pose,
            )
        if self.last_traj_points is not None:
            np.save(
                os.path.join(out_dir, f"{tag}_trajectory_points.npy"),
                self.last_traj_points,
            )

        if self.vis_gui:
            if self.heatmap_window is not None:
                self._refresh_window(self.heatmap_window)
                self.heatmap_window.capture_screen_image(
                    os.path.join(out_dir, f"{tag}_fisher_heatmap.png"),
                    do_render=True,
                )
            if self.velocity_window is not None:
                self._refresh_window(self.velocity_window)
                self.velocity_window.capture_screen_image(
                    os.path.join(out_dir, f"{tag}_fisher_velocity.png"),
                    do_render=True,
                )

        Log(
            (
                f"Saved {tag} Fisher artifacts under {out_dir}: "
                f"heatmap/velocity screenshots plus geometry snapshots"
            ),
            tag="NextBestView",
        )

    def export_frame0_artifacts_if_needed(self, idx: int):
        if idx != 0 or self.fisher_frame0_exported:
            return
        if self.last_gs_points is None or self.last_fisher_points is None:
            return

        out_dir = os.path.join(self.save_dir, "nbv_vis")
        os.makedirs(out_dir, exist_ok=True)

        merged_points = np.vstack([self.last_gs_points, self.last_fisher_points])
        merged_colors = np.vstack([self.last_gs_colors, self.last_fisher_colors])

        merged_pc = o3d.geometry.PointCloud()
        merged_pc.points = o3d.utility.Vector3dVector(merged_points)
        merged_pc.colors = o3d.utility.Vector3dVector(merged_colors)

        ply_path = os.path.join(out_dir, "frame0_gs_plus_fisher_hemi.ply")
        png_path = os.path.join(out_dir, "frame0_gs_plus_fisher_hemi.png")

        o3d.io.write_point_cloud(ply_path, merged_pc)
        screenshot_window = self.heatmap_window
        if self.vis_gui and screenshot_window is not None:
            screenshot_window.capture_screen_image(png_path, do_render=True)
            save_msg = f"Saved frame-0 Fisher artifacts: {ply_path}, {png_path}"
        else:
            save_msg = (
                f"Saved frame-0 Fisher artifacts (without screenshot): {ply_path}"
            )

        self.fisher_frame0_exported = True
        Log(save_msg, tag="NextBestView")
