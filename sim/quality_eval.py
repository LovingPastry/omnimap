"""Reconstruction quality probes for the simulation closed loop.

Two measurements, both independent of the policy under test:

* held-out PSNR -- a fixed set of hemisphere poses, chosen once before the run
  and never fed to the mapper; the simulator renders ground truth, the current
  3DGS map renders the prediction. Computed on pixels where ground truth has
  valid depth, so the black background does not inflate the score.
* surface completeness / accuracy -- fraction of ground-truth object points
  within ``tol_m`` of some gaussian centre (completeness), and fraction of
  gaussian centres inside the object region within ``tol_m`` of ground truth
  (accuracy).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from scipy.spatial import cKDTree

from omnimap.util.lookat import look_at_c2w


def heldout_poses(
    *,
    center: np.ndarray,
    radius: float,
    num_views: int = 16,
    phi_min_deg: float = 10.0,
    phi_max_deg: float = 80.0,
    theta_offset: float = 0.37,
) -> list[np.ndarray]:
    """Fibonacci directions in an elevation band, offset so none coincide with a start pose."""
    z_lo, z_hi = math.sin(math.radians(phi_min_deg)), math.sin(math.radians(phi_max_deg))
    golden = math.pi * (3.0 - math.sqrt(5.0))
    poses = []
    for i in range(num_views):
        z = z_lo + (z_hi - z_lo) * (i + 0.5) / num_views
        r = math.sqrt(max(1.0 - z * z, 0.0))
        theta = theta_offset + golden * i
        eye = np.asarray(center, dtype=np.float64) + radius * np.array(
            [r * math.cos(theta), r * math.sin(theta), z]
        )
        poses.append(look_at_c2w(eye=eye, target=np.asarray(center, dtype=np.float64)))
    return poses


def render_object_masks(simulator: Any, poses: list[np.ndarray]) -> list[np.ndarray]:
    """Object pixels per pose. Call before a ground plane is added to the simulator."""
    masks = []
    for c2w in poses:
        depth = simulator.render(c2w).depth
        masks.append(np.isfinite(depth) & (depth > 0))
    return masks


class QualityEvaluator:
    def __init__(
        self,
        *,
        simulator: Any,
        object_masks: list[np.ndarray] | None,
        policy: Any,
        intrinsics: np.ndarray,
        image_size: tuple[int, int],
        center: np.ndarray,
        radius: float,
        gt_points: np.ndarray,
        object_bounds: tuple[np.ndarray, np.ndarray],
        num_views: int = 16,
        tol_m: float = 0.01,
    ) -> None:
        self.policy = policy
        self.intrinsics = np.asarray(intrinsics, dtype=np.float64)
        self.image_size = image_size
        self.poses = heldout_poses(center=center, radius=radius, num_views=num_views)
        self.gt = []
        for k, c2w in enumerate(self.poses):
            res = simulator.render(c2w)
            rgb = torch.as_tensor(res.rgb, dtype=torch.float32) / 255.0
            # PSNR on object pixels only: with a ground plane in the scene, valid
            # depth would otherwise be dominated by the floor.
            valid = (
                object_masks[k]
                if object_masks is not None
                else (np.isfinite(res.depth) & (res.depth > 0))
            )
            self.gt.append((rgb.permute(2, 0, 1).contiguous(), torch.as_tensor(valid)))
        self.gt_tree = cKDTree(np.asarray(gt_points, dtype=np.float64))
        self.gt_points = np.asarray(gt_points, dtype=np.float64)
        self.lo, self.hi = (np.asarray(b, dtype=np.float64) for b in object_bounds)
        self.tol = float(tol_m)

    @torch.no_grad()
    def evaluate(self, gs_backend: Any) -> dict[str, float]:
        from omnimap.gaussian.renderer import render

        gaussians = gs_backend.gaussians
        background = gs_backend.background
        device = background.device
        psnrs = []
        for c2w, (gt_rgb, valid) in zip(self.poses, self.gt):
            cam = self.policy._build_tracking_camera_from_c2w(
                c2w=c2w,
                intrinsics_vec=self.intrinsics,
                image_size=self.image_size,
                idx=0,
                device=str(device),
            )
            pred = render(cam, gaussians, background)["render"].clamp(0.0, 1.0)
            gt_rgb_d = gt_rgb.to(device)
            mask = valid.to(device)
            if int(mask.sum()) == 0:
                continue
            mse = ((pred - gt_rgb_d) ** 2).mean(dim=0)[mask].mean().item()
            psnrs.append(10.0 * math.log10(1.0 / max(mse, 1e-12)))

        xyz = gaussians.get_xyz.detach().cpu().numpy().astype(np.float64)
        if xyz.shape[0] > 0:
            tree = cKDTree(xyz)
            d_gt, _ = tree.query(self.gt_points, k=1)
            completeness = float((d_gt <= self.tol).mean())
            inside = np.all((xyz >= self.lo) & (xyz <= self.hi), axis=1)
            if inside.any():
                d_pred, _ = self.gt_tree.query(xyz[inside], k=1)
                accuracy = float((d_pred <= self.tol).mean())
            else:
                accuracy = float("nan")
        else:
            completeness, accuracy = 0.0, float("nan")
        return {
            "heldout_psnr": float(np.mean(psnrs)) if psnrs else float("nan"),
            "heldout_psnr_min": float(np.min(psnrs)) if psnrs else float("nan"),
            "completeness": completeness,
            "accuracy": accuracy,
            "num_gaussians": int(xyz.shape[0]),
        }
