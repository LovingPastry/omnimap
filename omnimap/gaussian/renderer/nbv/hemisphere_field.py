from __future__ import annotations

import math
from typing import Dict, Tuple

import cv2
import numpy as np
import torch


def fibonacci_hemisphere_dirs(n: int, device: torch.device) -> torch.Tensor:
    i = torch.arange(n, device=device, dtype=torch.float32) + 0.5
    z = i / n
    golden = (1.0 + 5.0**0.5) / 2.0
    az = (2.0 * math.pi * i / golden) % (2.0 * math.pi)
    r = torch.sqrt(torch.clamp(1.0 - z * z, min=0.0))
    x = r * torch.cos(az)
    y = r * torch.sin(az)
    return torch.stack([x, y, z], dim=-1)


def scalarize_value(value) -> float:
    tensor = torch.as_tensor(value).detach().float()
    tensor = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
    return float(tensor.mean().item())


def idw_on_sphere(
    sample_dirs: torch.Tensor,
    sample_vals: torch.Tensor,
    query_dirs: torch.Tensor,
    power: float = 2.0,
) -> torch.Tensor:
    sample_vals = torch.as_tensor(sample_vals, dtype=torch.float32, device=sample_dirs.device)
    dots = torch.clamp(query_dirs @ sample_dirs.T, -1.0, 1.0)
    d = 1.0 - dots
    weights = 1.0 / (torch.pow(d, power) + 1e-6)
    weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
    if sample_vals.ndim == 1:
        return (weights * sample_vals[None, :]).sum(dim=1)
    return weights @ sample_vals


def dirs_to_theta_phi(dirs: torch.Tensor) -> torch.Tensor:
    dirs = torch.as_tensor(dirs, dtype=torch.float32)
    theta = torch.atan2(dirs[:, 1], dirs[:, 0])
    phi = torch.asin(torch.clamp(dirs[:, 2], 0.0, 1.0))
    return torch.stack([theta, phi], dim=1)


def angle_grad_to_tangent_dirs(
    theta_phi: torch.Tensor,
    grad_theta_phi: torch.Tensor,
) -> torch.Tensor:
    theta_phi = torch.as_tensor(theta_phi, dtype=torch.float32)
    grad_theta_phi = torch.as_tensor(
        grad_theta_phi, dtype=torch.float32, device=theta_phi.device
    )
    theta = theta_phi[:, 0]
    phi = theta_phi[:, 1]
    ct = torch.cos(theta)
    st = torch.sin(theta)
    cp = torch.cos(phi)
    sp = torch.sin(phi)

    e_theta = torch.stack([-cp * st, cp * ct, torch.zeros_like(theta)], dim=1)
    e_phi = torch.stack([-sp * ct, -sp * st, cp], dim=1)
    vel = grad_theta_phi[:, 0:1] * e_theta + grad_theta_phi[:, 1:2] * e_phi
    vel_norm = torch.linalg.norm(vel, dim=1, keepdim=True).clamp(min=1e-12)
    return (vel / vel_norm).to(torch.float32)


def fisher_values_to_colors(
    fisher_vals: torch.Tensor,
) -> Tuple[np.ndarray, torch.Tensor, Dict[str, float]]:
    fisher_vals = fisher_vals.detach().float()
    # fisher_vals = torch.log1p(torch.clamp(fisher_vals, min=0.0))
    lo = torch.quantile(fisher_vals, 0.05)
    hi = torch.quantile(fisher_vals, 0.95)
    denom = torch.clamp(hi - lo, min=1e-8)
    fisher_norm = torch.clamp((fisher_vals - lo) / denom, 0.0, 1.0)
    fisher_u8 = (fisher_norm * 255.0).to(torch.uint8).cpu().numpy()
    fisher_img = fisher_u8.reshape(-1, 1)
    bgr = cv2.applyColorMap(fisher_img, cv2.COLORMAP_TURBO).reshape(-1, 3)
    rgb = bgr[:, ::-1].astype(np.float32) / 255.0
    return (
        rgb,
        fisher_norm,
        {
            "lo": float(lo.item()),
            "hi": float(hi.item()),
            "denom": float(denom.item()),
            "transform": "linear",
            "raw_min": float(fisher_vals.min().item()),
            "raw_max": float(fisher_vals.max().item()),
        },
    )
