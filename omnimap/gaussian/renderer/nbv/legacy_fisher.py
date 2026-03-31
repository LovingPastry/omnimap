from __future__ import annotations

import copy
import math

import torch

from util.utils import Log
from gaussian.utils.camera_utils import Camera, HemisphereCamera
from gaussian.renderer.nbv.debug import build_debug_messages
from gaussian.renderer.nbv.hemisphere_field import (
    fibonacci_hemisphere_dirs,
    fisher_values_to_colors,
    idw_on_sphere,
    scalarize_value,
)
from gaussian.renderer.nbv.interfaces import FisherEvalResult, HemisphereFieldResult


class LegacyFisherEvaluator:
    def __init__(self, gaussians, config):
        self.gaussians = gaussians
        self.config = config
        self.reg_lambda = float(config.get("fisher_reg_lambda", 0.1))
        self.keyviewpoints = []

    def _zero_stat(self) -> torch.Tensor:
        params = [self.gaussians.capture()[1], self.gaussians.capture()[6]]
        return torch.zeros(
            sum(p.numel() for p in params),
            device=params[0].device,
            dtype=params[0].dtype,
        )

    def _compute_current_raw_grad(self, cam: Camera) -> torch.Tensor:
        return self.gaussians.cal_cur_hessian(cam=cam, return_per_point=False)

    def _current_stat_from_raw_grad(self, raw_grad: torch.Tensor) -> torch.Tensor:
        return raw_grad

    def _debug_labels(self):
        return {
            "stat_label": "cur_hessian",
            "history_label": "history_hessian",
            "denominator_label": "history_hessian + lambda",
            "contribution_label": "cur/(history+lambda)",
        }

    def compute_history_stat(self, keyviewpoints) -> torch.Tensor:
        if keyviewpoints is None:
            keyviewpoints = self.keyviewpoints
        if keyviewpoints is None or len(keyviewpoints) == 0:
            return self._zero_stat()

        history_stat = self._zero_stat()
        for keyframe in keyviewpoints:
            raw_grad = self._compute_current_raw_grad(keyframe)
            history_stat += self._current_stat_from_raw_grad(raw_grad)
        return history_stat

    def compute_view_score(
        self, cam: Camera, history_stat: torch.Tensor
    ) -> FisherEvalResult:
        raw_grad = self._compute_current_raw_grad(cam)
        cur_stat = self._current_stat_from_raw_grad(raw_grad)
        denominator = history_stat + self.reg_lambda
        contribution = cur_stat * torch.reciprocal(denominator)
        score = float(contribution.sum().item())
        topk_contrib_vals, _ = torch.topk(
            contribution.reshape(-1),
            k=min(10, contribution.numel()),
        )
        debug_stats = {
            "denominator": denominator.detach(),
            "contribution": contribution.detach(),
            "top10_contrib_ratio": float(
                topk_contrib_vals.sum().item() / max(score, 1e-8)
            ),
            "nonpositive_ratio": float((denominator <= 0).float().mean().item()),
        }
        return FisherEvalResult(
            score=score,
            stat=cur_stat.detach(),
            raw_grad=raw_grad.detach(),
            debug_stats=debug_stats,
        )

    def compute_view_gradient(
        self,
        hemisphere_cam: HemisphereCamera,
        history_stat: torch.Tensor,
        eps: float = 0.01,
    ) -> torch.Tensor:
        base_theta = float(hemisphere_cam.theta.item())
        base_phi = float(hemisphere_cam.phi.item())
        device = hemisphere_cam.theta.device

        def fisher_at(theta_val, phi_val):
            temp_cam = copy.deepcopy(hemisphere_cam)
            temp_cam.set_angles(theta_val, phi_val)
            fisher = self.compute_view_score(temp_cam, history_stat).score
            Log(
                f"Fisher at (theta={theta_val:.3f}, phi={phi_val:.3f}): {fisher:.6f}",
                tag="NextBestView",
            )
            return fisher

        f_theta_plus = fisher_at(base_theta + eps, base_phi)
        f_theta_minus = fisher_at(base_theta - eps, base_phi)
        dtheta = (f_theta_plus - f_theta_minus) / (2.0 * eps)

        phi_min, phi_max = 0.0, math.pi / 2.0
        phi_step_plus = min(eps, max(phi_max - base_phi, 0.0))
        phi_step_minus = min(eps, max(base_phi - phi_min, 0.0))

        if phi_step_plus > 0.0 and phi_step_minus > 0.0:
            f_phi_plus = fisher_at(base_theta, base_phi + phi_step_plus)
            f_phi_minus = fisher_at(base_theta, base_phi - phi_step_minus)
            dphi = (f_phi_plus - f_phi_minus) / (phi_step_plus + phi_step_minus)
        elif phi_step_plus > 0.0:
            f_phi_base = fisher_at(base_theta, base_phi)
            f_phi_plus = fisher_at(base_theta, base_phi + phi_step_plus)
            dphi = (f_phi_plus - f_phi_base) / phi_step_plus
        elif phi_step_minus > 0.0:
            f_phi_base = fisher_at(base_theta, base_phi)
            f_phi_minus = fisher_at(base_theta, base_phi - phi_step_minus)
            dphi = (f_phi_base - f_phi_minus) / phi_step_minus
        else:
            dphi = 0.0

        Log(
            f"Fisher gradient: dF/dtheta={dtheta:.6f}, dF/dphi={dphi:.6f}",
            tag="NextBestView",
        )
        return torch.tensor([dtheta, dphi], device=device, dtype=torch.float32)

    def build_hemisphere_field(
        self,
        viewpoint: Camera,
        scene_center: torch.Tensor,
        idx: int,
        num_samples: int,
        num_dense_points: int,
        power: float,
    ) -> HemisphereFieldResult:
        base_hemi = HemisphereCamera.from_camera(viewpoint, scene_center)
        history_stat = self.compute_history_stat(None)

        sample_dirs = fibonacci_hemisphere_dirs(num_samples, scene_center.device)
        fisher_vals = []
        for direction in sample_dirs:
            theta = torch.atan2(direction[1], direction[0])
            phi = torch.asin(torch.clamp(direction[2], 0.0, 1.0))
            hc = copy.deepcopy(base_hemi)
            hc.set_angles(theta=float(theta.item()), phi=float(phi.item()))
            fisher_vals.append(
                scalarize_value(self.compute_view_score(hc, history_stat).score)
            )

        sample_vals = torch.tensor(
            fisher_vals, device=scene_center.device, dtype=torch.float32
        )
        dense_dirs = fibonacci_hemisphere_dirs(num_dense_points, scene_center.device)
        dense_vals = idw_on_sphere(sample_dirs, sample_vals, dense_dirs, power=power)
        dense_colors, fisher_norm, color_stats = fisher_values_to_colors(dense_vals)
        current_result = self.compute_view_score(base_hemi, history_stat)
        labels = self._debug_labels()
        debug_messages = build_debug_messages(
            idx=idx,
            base_hemi=base_hemi,
            history_stat=history_stat,
            current_result=current_result,
            sample_dirs=sample_dirs,
            sample_vals=sample_vals,
            dense_vals=dense_vals,
            fisher_norm=fisher_norm,
            color_stats=color_stats,
            **labels,
        )
        return HemisphereFieldResult(
            idx=idx,
            base_hemi=base_hemi,
            history_stat=history_stat,
            sample_dirs=sample_dirs,
            sample_vals=sample_vals,
            dense_dirs=dense_dirs,
            dense_vals=dense_vals,
            dense_colors=torch.from_numpy(dense_colors),
            fisher_norm=fisher_norm,
            color_stats=color_stats,
            debug_stats={
                "messages": debug_messages,
                "current_result": current_result,
                "score_label": "Fisher hemisphere updated",
            },
        )
