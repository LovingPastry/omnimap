from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from sim.motion_policy import FisherMotionPolicy, resolve_step_scales
from sim.omnimap_runner import (
    OmniMapRunner,
    build_fisher_debug_config_overrides,
)
from sim.test_phase2_omnimap_runner import (
    load_phase1_triplets,
    load_tsdf_depth_max,
    validate_phase1_inputs,
)


def parse_args() -> argparse.Namespace:
    """Parse the Phase-3 probe CLI for one-step Fisher motion updates."""
    parser = argparse.ArgumentParser(
        description=(
            "Phase-3 Fisher debug probe: warm up OmniMap from saved Phase-1 RGBD frames, "
            "then compute exactly one Fisher-driven next pose.\n\n"
            "Use this entrypoint when you want to inspect three layers separately:\n"
            "1. the raw Fisher angle gradient\n"
            "2. the scaled/clipped controller output\n"
            "3. the final next pose written to disk"
        ),
        epilog=(
            "Examples:\n"
            "  python3 sim/test_phase3_motion_policy.py \\\n"
            "    --input_dir sim/sim_outputs/phase1_kettle_scaled \\\n"
            "    --config config/sim_rtabmap_config.yaml \\\n"
            "    --output sim/sim_outputs/phase3 \\\n"
            "    --show_fisher_arrows \\\n"
            "    --vis_gui\n\n"
            "  python3 sim/test_phase3_motion_policy.py \\\n"
            "    --input_dir sim/sim_outputs/phase1_kettle_scaled \\\n"
            "    --fisher_step_scale 0.01 \\\n"
            "    --fisher_arrow_length 0.04 \\\n"
            "    --terminate"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Directory containing Phase-1 outputs: view_*_rgb.png, *_depth.npy, *_c2w.npy",
    )
    parser.add_argument(
        "--config",
        default="config/sim_rtabmap_config.yaml",
        help="Path to the OmniMap yaml config used for the Phase-2/3 warm-up",
    )
    parser.add_argument(
        "--output",
        default="sim/sim_outputs/phase3",
        help="Directory to save the Phase-3 next pose and debug summary",
    )
    parser.add_argument("--fx", type=float, default=525.0, help="Camera focal length fx")
    parser.add_argument("--fy", type=float, default=525.0, help="Camera focal length fy")
    parser.add_argument("--cx", type=float, default=319.5, help="Camera principal point cx")
    parser.add_argument("--cy", type=float, default=239.5, help="Camera principal point cy")
    parser.add_argument(
        "--depth_scale",
        type=float,
        default=1000.0,
        help="OmniMap depth-scale convention parameter passed through to the runner",
    )
    parser.add_argument(
        "--max_depth",
        type=float,
        default=None,
        help="Optional depth clipping threshold in meters before frames are fed to OmniMap",
    )
    parser.add_argument(
        "--scene",
        type=str,
        default="room_0",
        help="Scene name passed through to OmniMap GUI branches",
    )
    parser.add_argument(
        "--vis_gui",
        action="store_true",
        help="Enable OmniMap / Fisher visualization windows during the warm-up run",
    )
    parser.add_argument(
        "--fisher_step_scale",
        type=float,
        default=0.03,
        help="Primary Fisher control scale applied to both theta and phi before clipping",
    )
    parser.add_argument(
        "--fisher_step_scale_theta",
        type=float,
        default=None,
        help="Optional theta-only override for the Fisher control scale",
    )
    parser.add_argument(
        "--fisher_step_scale_phi",
        type=float,
        default=None,
        help="Optional phi-only override for the Fisher control scale",
    )
    parser.add_argument(
        "--cartesian",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use Cartesian velocity-field control instead of the legacy angular controller",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.1,
        help="Time step in seconds used by the Cartesian velocity controller",
    )
    parser.add_argument(
        "--radial_gain",
        type=float,
        default=2.0,
        help="Radial correction gain used to pull the camera back toward the reference sphere",
    )
    parser.add_argument(
        "--fisher_arrow_length",
        type=float,
        default=0.07,
        help="Arrow length for Fisher velocity visualization; does not affect control",
    )
    parser.add_argument(
        "--show_fisher_heatmap",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show or hide the Fisher hemisphere heatmap in the GUI without changing control",
    )
    parser.add_argument(
        "--show_fisher_arrows",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show or hide the red velocity arrows; also enables/disables arrow computation",
    )
    parser.add_argument(
        "--fisher_debug_log",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Print additional Fisher velocity-field debug logs from the OmniMap side",
    )
    parser.add_argument(
        "--fisher_window_mode",
        choices=("combined", "split"),
        default="combined",
        help="Render Fisher heatmap and arrows in one window or split them into two windows",
    )
    parser.add_argument(
        "--fisher_heatmap_window_name",
        type=str,
        default="Fisher Heatmap Viewer",
        help="Open3D window title used for the Fisher heatmap window in split mode",
    )
    parser.add_argument(
        "--fisher_velocity_window_name",
        type=str,
        default="Fisher Velocity Viewer",
        help="Open3D window title used for the Fisher velocity window in split mode",
    )
    parser.add_argument(
        "--fisher_num_samples",
        type=int,
        default=64,
        help="Number of sparse hemisphere sample points used to compute Fisher values and gradients",
    )
    parser.add_argument(
        "--fisher_num_dense_points",
        type=int,
        default=4096,
        help="Number of dense hemisphere points used to interpolate both the colored Fisher field and the displayed velocity arrows",
    )
    parser.add_argument(
        "--fisher_idw_power",
        type=float,
        default=2.0,
        help="IDW interpolation power used for the dense Fisher heatmap",
    )
    parser.add_argument(
        "--fisher_display_radius_scale",
        type=float,
        default=0.92,
        help="Display radius scale for the dense Fisher heatmap relative to the true hemisphere radius",
    )
    parser.add_argument(
        "--fisher_arrow_radius_scale",
        type=float,
        default=0.90,
        help="Display radius scale for the velocity arrows relative to the true hemisphere radius",
    )
    parser.add_argument(
        "--grad_eps",
        type=float,
        default=0.01,
        help="Advanced: finite-difference epsilon used by the Fisher angle-gradient query",
    )
    parser.add_argument(
        "--max_delta_theta",
        type=float,
        default=0.20,
        help="Advanced: maximum allowed theta update per step in radians",
    )
    parser.add_argument(
        "--max_delta_phi",
        type=float,
        default=0.15,
        help="Advanced: maximum allowed phi update per step in radians",
    )
    parser.add_argument(
        "--fallback_delta_theta",
        type=float,
        default=0.03,
        help="Advanced: fallback theta increment used when the gradient norm is too small",
    )
    parser.add_argument(
        "--fallback_delta_phi",
        type=float,
        default=0.0,
        help="Advanced: fallback phi increment used when the gradient norm is too small",
    )
    parser.add_argument(
        "--terminate",
        action="store_true",
        help="Call omni.terminate() after the warm-up frames and policy step",
    )
    parser.add_argument(
        "--hold_gui_sec",
        type=float,
        default=2.0,
        help="Keep GUI windows open for a short time before the one-shot script exits",
    )
    return parser.parse_args()


def main() -> None:
    """Warm up OmniMap, query Fisher once, and persist the proposed next pose."""
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_dir = Path(args.input_dir)
    triplets = load_phase1_triplets(input_dir)
    tsdf_depth_max_m = load_tsdf_depth_max(args.config)
    validate_phase1_inputs(triplets, tsdf_depth_max_m=tsdf_depth_max_m)

    intrinsics = np.array([args.fx, args.fy, args.cx, args.cy], dtype=np.float32)
    step_scale_theta, step_scale_phi = resolve_step_scales(
        args.fisher_step_scale,
        theta_scale=args.fisher_step_scale_theta,
        phi_scale=args.fisher_step_scale_phi,
    )
    fisher_config_overrides = build_fisher_debug_config_overrides(
        show_fisher_heatmap=args.show_fisher_heatmap,
        show_fisher_arrows=args.show_fisher_arrows,
        fisher_arrow_length=args.fisher_arrow_length,
        fisher_debug_log=args.fisher_debug_log,
        fisher_window_mode=args.fisher_window_mode,
        fisher_heatmap_window_name=args.fisher_heatmap_window_name,
        fisher_velocity_window_name=args.fisher_velocity_window_name,
        fisher_num_samples=args.fisher_num_samples,
        fisher_num_dense_points=args.fisher_num_dense_points,
        fisher_idw_power=args.fisher_idw_power,
        fisher_display_radius_scale=args.fisher_display_radius_scale,
        fisher_arrow_radius_scale=args.fisher_arrow_radius_scale,
    )
    runner = OmniMapRunner.from_config_path(
        config_path=args.config,
        output=str(output_dir),
        depth_scale=args.depth_scale,
        vis_gui=args.vis_gui,
        scene=args.scene,
        max_depth_m=args.max_depth,
        config_overrides=fisher_config_overrides,
        verbose=True,
    )

    last_rgb_shape: tuple[int, int] | None = None
    last_c2w: np.ndarray | None = None
    last_idx: int | None = None

    # Phase 3 deliberately starts from saved Phase-1 outputs so the motion policy
    # can be validated before introducing the full closed-loop runtime.
    for order, (idx, rgb_path, depth_path, c2w_path) in enumerate(triplets):
        rgb_bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if rgb_bgr is None:
            raise RuntimeError(f"Failed to read RGB image: {rgb_path}")
        rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
        depth_m = np.load(depth_path).astype(np.float32)
        c2w = np.load(c2w_path).astype(np.float64)

        runner.step(
            idx=idx,
            rgb=rgb,
            depth_m=depth_m,
            c2w=c2w,
            intrinsics_vec=intrinsics,
            is_last=order == len(triplets) - 1,
        )
        last_rgb_shape = rgb.shape[:2]
        last_c2w = c2w
        last_idx = idx

    if last_rgb_shape is None or last_c2w is None or last_idx is None:
        raise RuntimeError("No valid warm-up frame was processed")

    policy = FisherMotionPolicy(
        step_gain_theta=step_scale_theta,
        step_gain_phi=step_scale_phi,
        cartesian=args.cartesian,
        dt=args.dt,
        radial_gain=args.radial_gain,
        grad_eps=args.grad_eps,
        max_delta_theta=args.max_delta_theta,
        max_delta_phi=args.max_delta_phi,
        fallback_delta_theta=args.fallback_delta_theta,
        fallback_delta_phi=args.fallback_delta_phi,
        verbose=True,
    )
    # This is the key Phase-3 handoff: current mapping state -> one Fisher-driven next pose.
    result = policy.next_pose_from_c2w(
        gs_backend=runner.omni.gs,
        current_c2w=last_c2w,
        intrinsics_vec=intrinsics,
        image_size=last_rgb_shape,
        idx=last_idx + 1,
    )

    np.save(output_dir / "phase3_next_c2w.npy", result.next_c2w)
    result_json = result.to_jsonable()
    result_json["fisher_controller"] = {
        "fisher_step_scale": float(args.fisher_step_scale),
        "step_scale_theta": float(step_scale_theta),
        "step_scale_phi": float(step_scale_phi),
        "cartesian": bool(args.cartesian),
        "dt": float(args.dt),
        "radial_gain": float(args.radial_gain),
        "grad_eps": float(args.grad_eps),
        "max_delta_theta": float(args.max_delta_theta),
        "max_delta_phi": float(args.max_delta_phi),
        "fallback_delta_theta": float(args.fallback_delta_theta),
        "fallback_delta_phi": float(args.fallback_delta_phi),
    }
    result_json["fisher_visualization"] = {
        "show_fisher_heatmap": bool(args.show_fisher_heatmap),
        "show_fisher_arrows": bool(args.show_fisher_arrows),
        "fisher_arrow_length": float(args.fisher_arrow_length),
        "fisher_debug_log": bool(args.fisher_debug_log),
        "fisher_window_mode": str(args.fisher_window_mode),
        "fisher_heatmap_window_name": str(args.fisher_heatmap_window_name),
        "fisher_velocity_window_name": str(args.fisher_velocity_window_name),
        "fisher_num_samples": int(args.fisher_num_samples),
        "fisher_num_dense_points": int(args.fisher_num_dense_points),
        "fisher_idw_power": float(args.fisher_idw_power),
        "fisher_display_radius_scale": float(args.fisher_display_radius_scale),
        "fisher_arrow_radius_scale": float(args.fisher_arrow_radius_scale),
    }
    with open(output_dir / "phase3_motion_result.json", "w", encoding="utf-8") as f:
        json.dump(result_json, f, indent=2)

    print(
        "[Phase3] Saved next pose to "
        f"{output_dir / 'phase3_next_c2w.npy'} and debug summary to "
        f"{output_dir / 'phase3_motion_result.json'}"
    )

    if args.vis_gui and args.hold_gui_sec > 0:
        import time

        print(f"[Phase3] Holding GUI for {args.hold_gui_sec:.2f}s before exit")
        time.sleep(float(args.hold_gui_sec))

    if args.terminate:
        runner.terminate()


if __name__ == "__main__":
    main()
