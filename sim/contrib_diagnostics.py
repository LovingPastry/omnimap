"""Where does the Fisher score of a view come from: observed or never-observed gaussians?

The runtime score of a view is ``sum_i w_i * cur_i / (history_i + lambda)`` over the
flattened ``[xyz(N*3), opacity(N)]`` parameter elements (``w`` = nbv_policy mask, view
gate included). An element that no keyframe has constrained has ``history_i ~ 0``, so
its denominator is just ``lambda`` (0.1) and its contribution is amplified ~10x more
than an element with ``history_i = 1``. If the large "information elsewhere" that keeps
the stop criterion from firing (global probe 1.3-180x the past peak) is mostly made of
such elements, the gap is an artefact of the denominator, not evidence of unseen
surface.

Every ``every`` steps this scores the current view plus ``num_dirs`` reachable
directions on the reference sphere (read-only: no map update), and records per view:

* the score split by history bucket (``[0, .1L)``, ``[.1L, L)``, ``[L, 10L)``, ``[10L, inf)``);
* the score re-weighted to other regularisers ``lambda'`` -- exactly
  ``sum_i contrib_i * (h_i + L) / (h_i + L')`` -- a counterfactual for "what if the
  denominator saturated";
* the angular distance to the nearest visited view direction.

Rows go to ``contrib_diag.jsonl``; ``sim/ablation/analyze_eps_contrib.py`` summarises them.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from omnimap.gaussian.renderer.nbv.motion_policy import _look_at_c2w
from omnimap.gaussian.renderer.nbv.policy_improvements import _fibonacci_band_dirs

LAMBDA_PRIMES = (0.1, 1.0, 10.0, 100.0)


class ContributionDiagnostics:
    def __init__(
        self,
        *,
        policy: Any,
        intrinsics_vec: Sequence[float],
        image_size: Sequence[int],
        out_path: Path,
        every: int = 30,
        num_dirs: int = 24,
        phi_min_deg: float = 15.0,
        phi_max_deg: float = 80.0,
    ) -> None:
        self.policy = policy
        self.intrinsics_vec = list(intrinsics_vec)
        self.image_size = tuple(image_size)
        self.every = int(every)
        self.dirs = _fibonacci_band_dirs(
            int(num_dirs), math.radians(phi_min_deg), math.radians(phi_max_deg)
        )
        self.visited: list[np.ndarray] = []
        self.out_path = Path(out_path)
        self._fh = open(self.out_path, "w", encoding="utf-8")

    def close(self) -> None:
        self._fh.close()

    def observe(self, camera_position: np.ndarray) -> None:
        """Record the view direction of every executed step (for the coverage split)."""
        if not self.policy.reference_initialized:
            return
        center = np.asarray(self.policy.reference_scene_center, dtype=np.float64)
        d = np.asarray(camera_position, dtype=np.float64).reshape(3) - center
        n = float(np.linalg.norm(d))
        if n > 1e-9:
            self.visited.append(d / n)

    def _nearest_visited_deg(self, direction: np.ndarray) -> float:
        if not self.visited:
            return 180.0
        cos = np.clip(np.stack(self.visited) @ direction, -1.0, 1.0)
        return float(np.degrees(np.arccos(cos.max())))

    @torch.no_grad()
    def _decompose(self, contribution: torch.Tensor, history: torch.Tensor, lam: float) -> dict:
        c = contribution.reshape(-1).double()
        h = history.reshape(-1).double()
        edges = (0.1 * lam, lam, 10.0 * lam)
        buckets = {
            "h_lt_0.1L": h < edges[0],
            "h_0.1L_L": (h >= edges[0]) & (h < edges[1]),
            "h_L_10L": (h >= edges[1]) & (h < edges[2]),
            "h_ge_10L": h >= edges[2],
        }
        total = float(c.sum().item())
        row = {"score": total}
        for name, sel in buckets.items():
            row[f"score_{name}"] = float(c[sel].sum().item())
        for lp in LAMBDA_PRIMES:
            row[f"score_lam{lp:g}"] = float((c * (h + lam) / (h + lp)).sum().item())
        return row

    def maybe_run(self, *, idx: int, gs_backend: Any, current_c2w: np.ndarray) -> dict | None:
        if self.every <= 0 or idx % self.every != 0 or not self.policy.reference_initialized:
            return None
        policy = self.policy
        fisher_eval = gs_backend.fisher_eval
        lam = float(fisher_eval.reg_lambda)
        center = np.asarray(policy.reference_scene_center, dtype=np.float64)
        radius = float(policy.reference_radius)
        device = policy._infer_runtime_device(gs_backend)

        policy._install_fisher_mask(gs_backend)
        try:
            history = fisher_eval.get_history_stat(
                getattr(gs_backend, "keyviewpoints", None), prefer_cached=True
            )
            h = history.reshape(-1)
            weight = fisher_eval.contribution_weight
            active = (weight.reshape(-1) > 0) if weight is not None and weight.numel() == h.numel() else torch.ones_like(h, dtype=torch.bool)
            n_active = max(int(active.sum().item()), 1)
            elem_frac = {
                "elem_frac_h_lt_0.1L": float(((h < 0.1 * lam) & active).sum().item()) / n_active,
                "elem_frac_h_lt_L": float(((h < lam) & active).sum().item()) / n_active,
            }

            def score_view(c2w: np.ndarray) -> dict:
                cam = policy._build_tracking_camera_from_c2w(
                    c2w=c2w,
                    intrinsics_vec=self.intrinsics_vec,
                    image_size=self.image_size,
                    idx=int(idx),
                    device=device,
                )
                result = fisher_eval.compute_view_score(cam, history)
                return self._decompose(result.debug_stats["contribution"], history, lam)

            cur_dir = np.asarray(current_c2w[:3, 3], dtype=np.float64) - center
            cur_dir = cur_dir / max(np.linalg.norm(cur_dir), 1e-12)
            current = score_view(np.asarray(current_c2w, dtype=np.float64))
            # Re-render the current direction on the reference sphere too, so the
            # comparison with the probes is at the same radius and look-at target.
            current_sphere = score_view(_look_at_c2w(eye=center + radius * cur_dir, target=center))
            probes = []
            for d in self.dirs:
                row = score_view(_look_at_c2w(eye=center + radius * d, target=center))
                row["dir"] = d.tolist()
                row["phi_deg"] = float(np.degrees(np.arcsin(np.clip(d[2], -1.0, 1.0))))
                row["nearest_visited_deg"] = self._nearest_visited_deg(d)
                probes.append(row)
        finally:
            if policy.fisher_mask is not None:
                policy._clear_fisher_mask(gs_backend)

        out = {
            "idx": int(idx),
            "lambda": lam,
            "num_gaussians": int(gs_backend.gaussians.get_xyz.shape[0]),
            "num_keyframes": len(getattr(gs_backend, "keyviewpoints", []) or []),
            "stop_max_window_mean": float(policy.stop_monitor.max_window_mean),
            "current": current,
            "current_sphere": current_sphere,
            "probes": probes,
            **elem_frac,
        }
        self._fh.write(json.dumps(out) + "\n")
        self._fh.flush()
        return out
