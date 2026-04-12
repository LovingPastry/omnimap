from __future__ import annotations

import math
from typing import Dict, List

import torch

try:
    from ...utils.camera_utils import HemisphereCamera
except ImportError:
    from gaussian.utils.camera_utils import HemisphereCamera


def summarize_tensor(name: str, tensor: torch.Tensor) -> str:
    values = torch.as_tensor(tensor).detach().float().reshape(-1)
    if values.numel() == 0:
        return f"{name}: empty"

    finite_mask = torch.isfinite(values)
    finite_count = int(finite_mask.sum().item())
    total_count = values.numel()
    if finite_count == 0:
        return f"{name}: no finite values ({total_count} entries)"

    finite_vals = values[finite_mask]
    quantiles = torch.quantile(
        finite_vals,
        torch.tensor([0.0, 0.05, 0.5, 0.95, 1.0], device=finite_vals.device),
    )
    neg_ratio = float((finite_vals < 0).float().mean().item())
    near_zero_ratio = float((finite_vals.abs() < 1e-8).float().mean().item())
    return (
        f"{name}: n={total_count}, finite={finite_count}/{total_count}, "
        f"mean={finite_vals.mean().item():.6e}, std={finite_vals.std(unbiased=False).item():.6e}, "
        f"min={quantiles[0].item():.6e}, p05={quantiles[1].item():.6e}, "
        f"p50={quantiles[2].item():.6e}, p95={quantiles[3].item():.6e}, "
        f"max={quantiles[4].item():.6e}, neg_ratio={neg_ratio:.2%}, "
        f"near_zero_ratio={near_zero_ratio:.2%}"
    )


def direction_to_angles_deg(direction: torch.Tensor):
    d = torch.as_tensor(direction).detach().float()
    d = d / torch.clamp(torch.linalg.norm(d), min=1e-8)
    theta = math.degrees(math.atan2(float(d[1].item()), float(d[0].item())))
    if theta < 0.0:
        theta += 360.0
    phi = math.degrees(math.asin(float(torch.clamp(d[2], 0.0, 1.0).item())))
    return theta, phi


def direction_angle_deg(a: torch.Tensor, b: torch.Tensor) -> float:
    a = torch.as_tensor(a).detach().float()
    b = torch.as_tensor(b).detach().float()
    a = a / torch.clamp(torch.linalg.norm(a), min=1e-8)
    b = b / torch.clamp(torch.linalg.norm(b), min=1e-8)
    cos_sim = torch.clamp(torch.dot(a, b), -1.0, 1.0)
    return math.degrees(math.acos(float(cos_sim.item())))


def build_debug_messages(
    *,
    idx: int,
    base_hemi: HemisphereCamera,
    history_stat: torch.Tensor,
    current_result,
    sample_dirs: torch.Tensor,
    sample_vals: torch.Tensor,
    dense_vals: torch.Tensor,
    fisher_norm: torch.Tensor,
    color_stats: Dict[str, float],
    stat_label: str,
    history_label: str,
    denominator_label: str,
    contribution_label: str,
) -> List[str]:
    score = float(current_result.score)
    stat = current_result.stat.detach().float()
    raw_grad = (
        current_result.raw_grad.detach().float()
        if current_result.raw_grad is not None
        else None
    )
    debug_stats = current_result.debug_stats or {}
    denom = debug_stats.get("denominator")
    contribution = debug_stats.get("contribution")
    top10_contrib_ratio = float(debug_stats.get("top10_contrib_ratio", 0.0))
    nonpositive_ratio = float(debug_stats.get("nonpositive_ratio", 0.0))

    theta_rad = float(base_hemi.theta.item())
    phi_rad = float(base_hemi.phi.item())
    theta_phi_dir = torch.tensor(
        [
            math.cos(phi_rad) * math.cos(theta_rad),
            math.cos(phi_rad) * math.sin(theta_rad),
            math.sin(phi_rad),
        ],
        device=sample_dirs.device,
        dtype=torch.float32,
    )
    theta_phi_dir = theta_phi_dir / torch.clamp(
        torch.linalg.norm(theta_phi_dir), min=1e-8
    )
    base_dir = (
        base_hemi.camera_center.detach().float() - base_hemi.center.detach().float()
    )
    base_dir = base_dir / torch.clamp(torch.linalg.norm(base_dir), min=1e-8)
    dir_consistency_angle = direction_angle_deg(theta_phi_dir, base_dir)
    expected_cam_pos = (
        base_hemi.center.detach().float() + base_hemi.radius * theta_phi_dir
    )
    cam_pos_residual = torch.linalg.norm(
        base_hemi.camera_center.detach().float() - expected_cam_pos
    ).item()

    sample_dots = torch.clamp(sample_dirs @ base_dir, -1.0, 1.0)
    nearest_idx = int(torch.argmax(sample_dots).item())
    nearest_angle = math.degrees(math.acos(float(sample_dots[nearest_idx].item())))
    nearest_val = float(sample_vals[nearest_idx].item())
    higher_count = int((sample_vals > sample_vals[nearest_idx]).sum().item())
    current_rank = higher_count + 1

    theta_phi_sample_dots = torch.clamp(sample_dirs @ theta_phi_dir, -1.0, 1.0)
    theta_phi_nearest_idx = int(torch.argmax(theta_phi_sample_dots).item())
    theta_phi_nearest_angle = math.degrees(
        math.acos(float(theta_phi_sample_dots[theta_phi_nearest_idx].item()))
    )
    theta_phi_nearest_val = float(sample_vals[theta_phi_nearest_idx].item())

    best_idx = int(torch.argmax(sample_vals).item())
    best_val = float(sample_vals[best_idx].item())
    best_angle = direction_angle_deg(sample_dirs[best_idx], base_dir)

    current_theta = math.degrees(float(base_hemi.theta.item()))
    current_phi = math.degrees(float(base_hemi.phi.item()))
    if current_theta < 0.0:
        current_theta += 360.0

    top_k = min(5, sample_vals.numel())
    top_vals, top_indices = torch.topk(sample_vals, k=top_k)
    top_entries = []
    for rank, (val, sample_idx) in enumerate(
        zip(top_vals.tolist(), top_indices.tolist()),
        start=1,
    ):
        theta_deg, phi_deg = direction_to_angles_deg(sample_dirs[sample_idx])
        angle_deg = direction_angle_deg(sample_dirs[sample_idx], base_dir)
        top_entries.append(
            f"#{rank}[idx={sample_idx}]={val:.6f}, angle_to_current={angle_deg:.2f}deg, "
            f"theta={theta_deg:.1f}deg, phi={phi_deg:.1f}deg"
        )

    messages = [
        (
            f"Fisher debug frame {idx}: current(theta={current_theta:.1f}deg, "
            f"phi={current_phi:.1f}deg, radius={base_hemi.radius:.4f}), "
            f"api_current={score:.6f}, exact_current={score:.6f}, "
            f"top10_contrib_ratio={top10_contrib_ratio:.2%}, "
            f"nearest_sample[idx={nearest_idx}]={nearest_val:.6f}, "
            f"current_rank={current_rank}/{sample_vals.numel()}, "
            f"nearest_angle={nearest_angle:.2f}deg, best_sample[idx={best_idx}]={best_val:.6f}, "
            f"best_angle_to_current={best_angle:.2f}deg"
        ),
        summarize_tensor(history_label, history_stat),
    ]
    if raw_grad is not None:
        messages.append(summarize_tensor("raw_grad(current_view)", raw_grad))
    messages.append(summarize_tensor(f"{stat_label}(current_view)", stat))
    if denom is not None:
        messages.append(
            summarize_tensor(denominator_label, denom)
            + f", nonpositive_ratio={nonpositive_ratio:.2%}"
        )
    if contribution is not None:
        messages.append(summarize_tensor(contribution_label, contribution))
        messages.extend(
            [
                summarize_tensor("sample_fisher", sample_vals),
                summarize_tensor("dense_fisher", dense_vals),
                (
                    f"Direction consistency: angle(theta_phi_dir vs camera_center_dir)="
                    f"{dir_consistency_angle:.4f}deg, "
                    f"camera_center_residual={cam_pos_residual:.6e}, "
                    f"nearest_by_theta_phi[idx={theta_phi_nearest_idx}]={theta_phi_nearest_val:.6f}, "
                    f"nearest_theta_phi_angle={theta_phi_nearest_angle:.2f}deg"
                ),
                (
                    f"Color normalization ({color_stats.get('transform', 'linear')} domain): "
                    f"lo={color_stats['lo']:.6f}, hi={color_stats['hi']:.6f}, "
                    f"denom={color_stats['denom']:.6f}, "
                    f"raw_min={color_stats.get('raw_min', float('nan')):.6f}, "
                    f"raw_max={color_stats.get('raw_max', float('nan')):.6f}"
                ),
                summarize_tensor("fisher_norm", fisher_norm),
                "Top Fisher sample directions: " + " | ".join(top_entries),
            ]
        )
    return messages
