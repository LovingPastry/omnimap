from __future__ import annotations

import os
from typing import Optional

import numpy as np
import open3d as o3d

from util.utils import Log
from gaussian.utils.sh_utils import SH2RGB


class FisherVisualizer:
    def __init__(self, gaussians, config, save_dir: str, vis_gui: bool):
        self.gaussians = gaussians
        self.config = config
        self.save_dir = save_dir
        self.vis_gui = vis_gui
        self.o3d_window = None
        self.fisher_hemi_geometry = None
        self.fisher_frame0_exported = False
        self.last_gs_points = None
        self.last_gs_colors = None
        self.last_fisher_points = None
        self.last_fisher_colors = None

    def attach_window(self, o3d_window):
        self.o3d_window = o3d_window

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

    def apply_field_result(self, field_result):
        center = field_result.base_hemi.center.detach().float()
        radius = float(field_result.base_hemi.radius)
        dense_points = center[None, :] + radius * field_result.dense_dirs

        hemi_pc = o3d.geometry.PointCloud()
        hemi_pc.points = o3d.utility.Vector3dVector(dense_points.detach().cpu().numpy())
        hemi_pc.colors = o3d.utility.Vector3dVector(
            field_result.dense_colors.cpu().numpy()
        )

        if self.vis_gui and self.o3d_window is not None:
            if self.fisher_hemi_geometry is not None:
                self.o3d_window.remove_geometry(
                    self.fisher_hemi_geometry, reset_bounding_box=False
                )
            self.fisher_hemi_geometry = hemi_pc
            self.o3d_window.add_geometry(
                self.fisher_hemi_geometry, reset_bounding_box=False
            )
            self.o3d_window.poll_events()
            self.o3d_window.update_renderer()

        self.last_fisher_points = np.asarray(hemi_pc.points).copy()
        self.last_fisher_colors = np.asarray(hemi_pc.colors).copy()
        if self.last_gs_points is None or self.last_gs_colors is None:
            self.update_gaussian_cache()
        return hemi_pc

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
        if self.vis_gui and self.o3d_window is not None:
            self.o3d_window.capture_screen_image(png_path, do_render=True)
            save_msg = f"Saved frame-0 Fisher artifacts: {ply_path}, {png_path}"
        else:
            save_msg = (
                f"Saved frame-0 Fisher artifacts (without screenshot): {ply_path}"
            )

        self.fisher_frame0_exported = True
        Log(save_msg, tag="NextBestView")
