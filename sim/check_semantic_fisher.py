"""Audit a real closed-loop run and its serialized Gaussian snapshot on GPU."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "omnimap"):
    sys.path.insert(0, str(path))

from info_flow.planner_snapshot import load_planner_snapshot_file
from omnimap.gaussian.renderer.nbv.motion_policy import FisherMotionPolicy
from omnimap.gaussian.renderer.nbv.policy_improvements import FisherContributionMask, NBVPolicyConfig


def check(run_dir: Path) -> dict:
    rows = [json.loads(line) for line in (run_dir / "loop_log.jsonl").read_text().splitlines()]
    if not rows:
        raise AssertionError("closed-loop log is empty")
    ready = [row for row in rows if row["nbv_semantic_status"] == "ready"]
    if not ready:
        raise AssertionError("no target was ever selected in the closed loop")
    for row in rows:
        if row["nbv_semantic_frame_idx"] != row["planning_snapshot_frame"]:
            raise AssertionError("semantic frame and planning geometry frame differ")
        if row["nbv_mask_kept_ratio"] == 0:
            np.testing.assert_array_equal(row["velocity_world"], np.zeros(3))
            np.testing.assert_array_equal(row["angular_velocity_world"], np.zeros(3))
            assert row["fisher_current_score"] == 0
    assert all(0 < row["nbv_mask_kept_ratio"] < 1 for row in ready)
    assert any(row["linear_speed_applied"] > 0 for row in ready)

    snapshot = load_planner_snapshot_file(run_dir / "planner_snapshot.pt")
    backend = snapshot.backend
    state = backend.gaussian_semantics
    assert state["frame_idx"] == snapshot.keyframe_idx
    assert not hasattr(backend, "tsdfs")
    pose = np.load(run_dir / "frames" / f"step_{snapshot.keyframe_idx:04d}_c2w.npy")
    cfg = NBVPolicyConfig.from_dict(backend.config["nbv_policy"])
    mask = FisherContributionMask(cfg.fisher_mask)
    weights = mask.gaussian_weights(backend.gaussians.get_xyz, semantic_state=state)
    expanded = mask.expand_to_contribution(weights)
    n = len(weights)
    assert 0 < weights.sum() < n

    def policy_step():
        policy = FisherMotionPolicy(
            fisher_step_scale=.03, cartesian=True, dt=.1, linear_vel_max=.02,
            nbv_config=backend.config["nbv_policy"], verbose=False,
        )
        return policy.next_pose_from_c2w(
            gs_backend=backend, current_c2w=pose,
            intrinsics_vec=[525., 525., 319.5, 239.5], image_size=(480, 640), idx=100,
        )

    baseline = policy_step()
    evaluator = backend.fisher_eval
    # The CUDA rasterizer still renders the complete scene. Increase only the
    # non-target Fisher entries by 1000x on each probe; selected score/gradient
    # must remain unchanged, even if excluded entries dominate the full score.
    original_raw = evaluator._compute_current_raw_grad

    def amplified_background(cam):
        raw = original_raw(cam)
        return raw * torch.where(expanded.bool(), 1., 1000.).to(raw)

    evaluator._compute_current_raw_grad = amplified_background
    amplified = policy_step()
    evaluator._compute_current_raw_grad = original_raw
    np.testing.assert_allclose(
        [baseline.fisher_score, baseline.grad_theta_raw, baseline.grad_phi_raw],
        [amplified.fisher_score, amplified.grad_theta_raw, amplified.grad_phi_raw],
        rtol=1e-5, atol=1e-3,
    )
    np.testing.assert_allclose(baseline.next_c2w, amplified.next_c2w, atol=1e-7)

    policy = FisherMotionPolicy(nbv_config=backend.config["nbv_policy"], verbose=False)
    cam = policy._build_tracking_camera_from_c2w(
        c2w=pose, intrinsics_vec=[525., 525., 319.5, 239.5],
        image_size=(480, 640), idx=100, device=backend.get_runtime_device(),
    )
    history = evaluator.get_history_stat(backend.keyviewpoints)
    unmasked = evaluator.compute_view_score(cam, history)
    evaluator.set_score_mask(expanded)
    masked = evaluator.compute_view_score(cam, history)
    evaluator.set_score_mask(None)
    assert torch.count_nonzero(masked.debug_stats["contribution"][expanded == 0]) == 0
    expected = float((unmasked.debug_stats["contribution"] * expanded).sum().item())
    np.testing.assert_allclose(masked.score, expected, rtol=1e-5)
    report = {
        "run_dir": str(run_dir), "steps": len(rows), "target_ready_steps": len(ready),
        "first_target_step": ready[0]["idx"],
        "target_ratio_range": [min(row["nbv_mask_kept_ratio"] for row in ready),
                               max(row["nbv_mask_kept_ratio"] for row in ready)],
        "initial_gaussians": rows[0]["num_gaussians"], "final_gaussians": n,
        "snapshot_frame": snapshot.keyframe_idx, "semantic_frame": state["frame_idx"],
        "selected_gaussians": int(weights.sum().item()), "excluded_gaussians": int((weights == 0).sum().item()),
        "final_target_instance_ids": torch.unique(state["instance_ids"][weights.bool()]).cpu().tolist(),
        "masked_score": masked.score, "unmasked_score": unmasked.score,
        "background_amplification": 1000, "masked_gradient_unchanged": True,
        "excluded_contributions_all_zero": True, "no_live_tsdf_in_planning": True,
        "path_length_m": sum(row["linear_speed_applied"] * row["dt"] for row in rows),
        "limitations": "One short single-view RGB-D-derived simulation; does not establish segmentation accuracy, reconstruction improvement, or robot performance.",
    }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = check(args.run_dir)
    output = args.output or args.run_dir / "semantic_validation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
