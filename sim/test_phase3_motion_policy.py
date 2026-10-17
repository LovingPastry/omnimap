from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from sim.motion_policy import FisherMotionPolicy
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
            "Phase-3 Fisher single-step probe: warm up OmniMap from saved Phase-1 RGBD "
            "frames, then compute exactly one Fisher-driven next pose.\n\n"
            "It uses the same controller as Phase 4, but only runs one control step, "
            "so it is the fastest way to inspect raw gradients, linear velocity, angular "
            "velocity, and the final integrated next pose."
        ),
        epilog=(
            "Recommended single-step Cartesian debug:\n"
            "  python3 sim/test_phase3_motion_policy.py \\\n"
            "    --input_dir sim/sim_outputs/phase1_kettle_scaled \\\n"
            "    --config config/sim_rtabmap_config.yaml \\\n"
            "    --output sim/sim_outputs/phase3_cartesian \\\n"
            "    --cartesian \\\n"
            "    --dt 0.1 \\\n"
            "    --fisher_step_scale 1e-4 \\\n"
            "    --linear_vel_max 0.5 \\\n"
            "    --radial_gain 0.2 \\\n"
            "    --angular_gain 2.0 \\\n"
            "    --show_fisher_arrows \\\n"
            "    --fisher_window_mode split \\\n"
            "    --fisher_num_samples 128 \\\n"
            "    --fisher_num_dense_points 1024 \\\n"
            "    --vis_gui \\\n"
            "    --hold_gui_sec 5\n\n"
            "Use this script when you want one-step numbers before running the full "
            "closed-loop entrypoint sim/sim_fisher_closed_loop.py."
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
        "--linear_vel_max",
        type=float,
        default=0.5,
        help="Maximum Cartesian linear speed used to clip the final velocity command in cartesian mode",
    )
    parser.add_argument(
        "--angular_gain",
        type=float,
        default=2.0,
        help="Angular gain applied to the pose-error rotvec when cartesian mode computes omega commands",
    )
    parser.add_argument(
        "--enable_angular",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable or disable angular velocity output in cartesian mode",
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
        "--spherical_speed_min",
        type=float,
        default=1e-4,
        help="Advanced: minimum spherical-speed norm; below this the controller stops instead of moving",
    )
    parser.add_argument(
        "--max_delta_theta",
        type=float,
        default=0.20,
        help="Advanced: theta component used to define the spherical-speed clip radius in radians per step",
    )
    parser.add_argument(
        "--max_delta_phi",
        type=float,
        default=0.15,
        help="Advanced: phi component used to define the spherical-speed clip radius in radians per step",
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
    step_scale = float(args.fisher_step_scale)
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
        fisher_step_scale=step_scale,
        cartesian=args.cartesian,
        dt=args.dt,
        radial_gain=args.radial_gain,
        linear_vel_max=args.linear_vel_max,
        angular_gain=args.angular_gain,
        enable_angular=args.enable_angular,
        grad_eps=args.grad_eps,
        spherical_speed_min=args.spherical_speed_min,
        # NOTE: CLI 名保持 --max_delta_theta/--max_delta_phi，构造参数已改名。
        max_theta_rate=args.max_delta_theta,
        max_phi_rate=args.max_delta_phi,
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
        "step_scale_theta": float(step_scale),
        "step_scale_phi": float(step_scale),
        "cartesian": bool(args.cartesian),
        "dt": float(args.dt),
        "radial_gain": float(args.radial_gain),
        "linear_vel_max": float(args.linear_vel_max),
        "angular_gain": float(args.angular_gain),
        "enable_angular": bool(args.enable_angular),
        "grad_eps": float(args.grad_eps),
        "spherical_speed_min": float(args.spherical_speed_min),
        "max_delta_theta": float(args.max_delta_theta),
        "max_delta_phi": float(args.max_delta_phi),
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
    csv_path = output_dir / "phase3_debug.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "idx",
                "controller_mode",
                "cartesian",
                "fisher_step_scale",
                "dt",
                "radial_gain",
                "linear_vel_max",
                "angular_gain",
                "enable_angular",
                "grad_theta_raw",
                "grad_phi_raw",
                "grad_theta_compressed",
                "grad_phi_compressed",
                "scaled_theta",
                "scaled_phi",
                "theta_rate_applied",
                "phi_rate_applied",
                "speed_clipped",
                "clip_scale_ratio",
                "grad_norm_raw",
                "grad_norm_compressed",
                "fisher_score",
                "spherical_speed_raw",
                "spherical_speed_scaled",
                "spherical_speed_applied",
                "spherical_speed_limit",
                "spherical_speed_min",
                "reference_radius",
                "current_radius",
                "radial_error",
                "vt_world_norm",
                "vn_world_norm",
                "velocity_raw_world_norm",
                "velocity_world_norm",
                "linear_speed_raw",
                "linear_speed_applied",
                "linear_speed_limit",
                "angular_speed_raw",
                "angular_speed_applied",
                "rotvec_error_norm",
                "angular_velocity_world_norm",
                "max_scale_before_clip",
                "should_stop",
                "stop_reason",
            ],
        )
        writer.writeheader()
        max_scale_before_clip = (
            float("nan")
            if args.cartesian
            else float(np.hypot(args.max_delta_theta, args.max_delta_phi))
            / max(result.grad_norm_raw, 1e-12)
        )
        writer.writerow(
            {
                "idx": result.idx,
                "controller_mode": result.controller_mode,
                "cartesian": result.cartesian,
                "fisher_step_scale": float(args.fisher_step_scale),
                "dt": float(args.dt),
                "radial_gain": float(args.radial_gain),
                "linear_vel_max": float(args.linear_vel_max),
                "angular_gain": float(args.angular_gain),
                "enable_angular": bool(args.enable_angular),
                "grad_theta_raw": result.grad_theta_raw,
                "grad_phi_raw": result.grad_phi_raw,
                "grad_theta_compressed": result.grad_theta_compressed,
                "grad_phi_compressed": result.grad_phi_compressed,
                "scaled_theta": result.scaled_theta,
                "scaled_phi": result.scaled_phi,
                "theta_rate_applied": result.theta_rate_applied,
                "phi_rate_applied": result.phi_rate_applied,
                "speed_clipped": result.speed_clipped,
                "clip_scale_ratio": result.clip_scale_ratio,
                "grad_norm_raw": result.grad_norm_raw,
                "grad_norm_compressed": result.grad_norm_compressed,
                "fisher_score": result.fisher_score,
                "spherical_speed_raw": result.spherical_speed_raw,
                "spherical_speed_scaled": result.spherical_speed_scaled,
                "spherical_speed_applied": result.spherical_speed_applied,
                "spherical_speed_limit": result.spherical_speed_limit,
                "spherical_speed_min": result.spherical_speed_min,
                "reference_radius": result.reference_radius,
                "current_radius": result.current_radius,
                "radial_error": result.radial_error,
                "vt_world_norm": float(np.linalg.norm(result.vt_world)),
                "vn_world_norm": float(np.linalg.norm(result.vn_world)),
                "velocity_raw_world_norm": float(np.linalg.norm(result.velocity_raw_world)),
                "velocity_world_norm": float(np.linalg.norm(result.velocity_world)),
                "linear_speed_raw": result.linear_speed_raw,
                "linear_speed_applied": result.linear_speed_applied,
                "linear_speed_limit": result.linear_speed_limit,
                "angular_speed_raw": result.angular_speed_raw,
                "angular_speed_applied": result.angular_speed_applied,
                "rotvec_error_norm": float(np.linalg.norm(result.rotvec_error)),
                "angular_velocity_world_norm": float(
                    np.linalg.norm(result.angular_velocity_world)
                ),
                "max_scale_before_clip": max_scale_before_clip,
                "should_stop": result.should_stop,
                "stop_reason": result.stop_reason,
            }
        )

    print(
        "[Phase3] Saved next pose to "
        f"{output_dir / 'phase3_next_c2w.npy'} and debug summary to "
        f"{output_dir / 'phase3_motion_result.json'}; CSV debug row to {csv_path}"
    )

    if args.vis_gui and args.hold_gui_sec > 0:
        import time

        print(f"[Phase3] Holding GUI for {args.hold_gui_sec:.2f}s before exit")
        time.sleep(float(args.hold_gui_sec))

    if args.terminate:
        runner.terminate()


if __name__ == "__main__":
    main()
