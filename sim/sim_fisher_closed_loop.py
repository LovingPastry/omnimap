from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from omnimap.util.utils import (
    configure_logging,
    get_section_logger,
    should_log_step,
)

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


def look_at_c2w(
    eye: np.ndarray,
    target: np.ndarray,
    up: np.ndarray | None = None,
) -> np.ndarray:
    """Build a `c2w` pose whose optical axis points from `eye` to `target`."""
    eye = np.asarray(eye, dtype=np.float64).reshape(3)
    target = np.asarray(target, dtype=np.float64).reshape(3)
    up = (
        np.array([0.0, 0.0, 1.0], dtype=np.float64)
        if up is None
        else np.asarray(up, dtype=np.float64).reshape(3)
    )

    forward = target - eye
    forward_norm = np.linalg.norm(forward)
    if forward_norm < 1e-12:
        raise ValueError("eye and target are too close; cannot build look-at pose")
    forward = forward / forward_norm

    right = np.cross(forward, up)
    right_norm = np.linalg.norm(right)
    if right_norm < 1e-12:
        fallback_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        right = np.cross(forward, fallback_up)
        right_norm = np.linalg.norm(right)
        if right_norm < 1e-12:
            raise ValueError("failed to construct a valid right axis for look-at pose")
    right = right / right_norm

    true_up = np.cross(right, forward)
    true_up = true_up / max(np.linalg.norm(true_up), 1e-12)
    down = -true_up

    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, 0] = right
    c2w[:3, 1] = down
    c2w[:3, 2] = forward
    c2w[:3, 3] = eye
    return c2w


def spherical_c2w(
    center: np.ndarray,
    radius: float,
    theta: float,
    phi: float,
) -> np.ndarray:
    """Convert a hemisphere `(radius, theta, phi)` state into a `c2w` pose."""
    center = np.asarray(center, dtype=np.float64).reshape(3)
    x = radius * np.cos(phi) * np.cos(theta)
    y = radius * np.cos(phi) * np.sin(theta)
    z = radius * np.sin(phi)
    eye = center + np.array([x, y, z], dtype=np.float64)
    return look_at_c2w(eye=eye, target=center)


def save_render_artifacts(
    *,
    save_dir: Path,
    idx: int,
    rgb: np.ndarray,
    depth: np.ndarray,
    c2w: np.ndarray,
) -> None:
    """Persist one closed-loop step's RGBD render and authoritative pose."""
    frame_prefix = save_dir / "frames" / f"step_{idx:04d}"
    frame_prefix.parent.mkdir(parents=True, exist_ok=True)

    rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(frame_prefix.with_name(frame_prefix.name + "_rgb.png")), rgb_bgr)
    np.save(frame_prefix.with_name(frame_prefix.name + "_depth.npy"), depth)
    np.save(frame_prefix.with_name(frame_prefix.name + "_c2w.npy"), c2w)

    valid = np.isfinite(depth) & (depth > 0)
    if np.any(valid):
        d_min = float(depth[valid].min())
        d_max = float(depth[valid].max())
        denom = max(d_max - d_min, 1e-6)
        norm = np.clip((depth - d_min) / denom, 0.0, 1.0)
    else:
        norm = np.zeros_like(depth, dtype=np.float32)
    vis = (norm * 255.0).astype(np.uint8)
    cv2.imwrite(
        str(frame_prefix.with_name(frame_prefix.name + "_depth_vis.png")),
        vis,
    )


def compute_depth_stats(depth: np.ndarray) -> tuple[float, float, int, int, float]:
    """Return `(min, max, valid_count, total_count, valid_ratio)` for one rendered depth map."""
    valid = np.isfinite(depth) & (depth > 0)
    total = int(depth.size)
    valid_count = int(valid.sum())
    valid_ratio = float(valid_count / max(total, 1))
    if valid_count == 0:
        return 0.0, 0.0, 0, total, valid_ratio
    return (
        float(depth[valid].min()),
        float(depth[valid].max()),
        valid_count,
        total,
        valid_ratio,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the Phase-4 closed-loop experiment CLI."""
    parser = argparse.ArgumentParser(
        description=(
            "Phase-4 closed-loop Fisher debug entrypoint: render RGBD from a static scene, "
            "feed frames into OmniMap, query Fisher gradients, and advance the camera "
            "for multiple steps.\n\n"
            "In cartesian mode, this entrypoint now uses the full control chain:\n"
            "linear velocity integration + angular velocity error -> omega command -> "
            "rotation integration. This is the authoritative runtime for inspecting "
            "whether the Fisher field, velocity field, mapping state, and camera "
            "trajectory remain consistent over time."
        ),
        epilog="""
# 推荐默认闭环命令：
python3 sim/sim_fisher_closed_loop.py \
    --pcd_path replica/log_pcd_0/对比结果/RoboSeg_geo/Kettle.ply \
    --point_scale 0.001 \
    --config config/sim_rtabmap_config.yaml \
    --num_steps 50 \
    --save_dir sim/sim_outputs/phase4_dense \
    --vis_gui \
    --show_fisher_arrows \
    --fisher_window_mode split \
    --fisher_num_samples 128 \
    --fisher_num_dense_points 1024 \
    --step_delay_sec 0.1 \
    --hold_gui_sec 2.0 \
    --cartesian \
    --dt 0.1 \
    --fisher_step_scale 1e-4 \
    --linear_vel_max 0.5 \
    --radial_gain 0.2 \
    --angular_gain 2.0

# 保存每一步渲染结果：
python3 sim/sim_fisher_closed_loop.py \
    --pcd_path replica/log_pcd_0/对比结果/RoboSeg_geo/Kettle.ply \
    --point_scale 0.001 \
    --config config/sim_rtabmap_config.yaml \
    --num_steps 10 \
    --save_dir sim/sim_outputs/phase4_frames \
    --save_frames \
    --vis_gui \
    --show_fisher_arrows
""",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--pcd_path", required=True, help="Path to a .ply or .pcd point cloud"
    )
    parser.add_argument("--config", default="config/sim_rtabmap_config.yaml")
    parser.add_argument("--save_dir", default="sim/sim_outputs/phase4")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fx", type=float, default=525.0)
    parser.add_argument("--fy", type=float, default=525.0)
    parser.add_argument("--cx", type=float, default=319.5)
    parser.add_argument("--cy", type=float, default=239.5)
    parser.add_argument("--voxel_size", type=float, default=None)
    parser.add_argument(
        "--point_scale",
        type=float,
        default=0.001,
        help=(
            "Global scale applied to point-cloud coordinates before simulation. "
            "Replica log point clouds are typically in millimeters, so 0.001 is the safe default."
        ),
    )
    parser.add_argument("--ground", action="store_true")
    parser.add_argument("--ground_size", type=float, default=1.0)
    parser.add_argument("--ground_z", type=float, default=0.0)
    parser.add_argument("--coord_frame", action="store_true")
    parser.add_argument("--scene", type=str, default="room_0")
    parser.add_argument("--depth_scale", type=float, default=1000.0)
    parser.add_argument("--max_depth", type=float, default=None)
    parser.add_argument("--num_steps", type=int, default=120)
    parser.add_argument("--hemi_radius", type=float, default=None)
    parser.add_argument("--radius_scale", type=float, default=1.5)
    parser.add_argument("--init_theta", type=float, default=0.0)
    parser.add_argument("--init_phi", type=float, default=0.35)
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
        "--angular_speed_max",
        type=float,
        default=1.0,
        help="Maximum norm of cartesian angular velocity command (rad/s)",
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
        default=128,
        help="Number of sparse hemisphere sample points used to compute Fisher values and gradients",
    )
    parser.add_argument(
        "--fisher_num_dense_points",
        type=int,
        default=1024,
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
    parser.add_argument("--vis_gui", action="store_true")
    parser.add_argument(
        "--headless",
        action="store_true",
        help=(
            "Run in explicit headless mode: force-disable GUI branches and set "
            "Open3D offscreen-friendly environment defaults when not already set."
        ),
    )
    parser.add_argument(
        "--save_frames",
        action="store_true",
        help="Save per-step RGB / depth / c2w artifacts under save_dir/frames",
    )
    parser.add_argument(
        "--step_delay_sec",
        type=float,
        default=0.0,
        help="Optional delay after each step, useful when watching GUI updates",
    )
    parser.add_argument(
        "--hold_gui_sec",
        type=float,
        default=0.0,
        help="Optional delay before process exit so GUI windows remain visible for a bit",
    )
    parser.add_argument(
        "--log_profile",
        choices=("quiet", "default", "debug"),
        default="default",
        help="Console logging profile. File log remains more verbose by default.",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default=None,
        help="Optional explicit logging level override (e.g., DEBUG/INFO/WARNING/ERROR).",
    )
    parser.add_argument(
        "--log_section",
        action="append",
        choices=("all", "main", "tsdf", "gaussian", "fisher", "planner", "profile"),
        default=None,
        help="选择输出日志分区；可重复传入。未指定时默认 all。",
    )
    parser.add_argument(
        "--log_min_level",
        choices=("DEBUG", "INFO", "WARNING"),
        default="INFO",
        help="终端最小日志等级阈值。",
    )
    parser.add_argument(
        "--log_every",
        type=int,
        default=10,
        help="Emit per-step summary every N steps (and always on stop/final step).",
    )
    parser.add_argument(
        "--log_file",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable or disable run.log output in save_dir.",
    )
    parser.add_argument("--terminate", action="store_true")
    return parser.parse_args(argv)


def run_closed_loop(args: argparse.Namespace) -> None:
    """Run the authoritative Phase-4 loop from render to next-pose update."""
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    log_path_file = str(save_dir / "run.log") if bool(args.log_file) else None
    requested_sections = args.log_section or ["all"]
    selected_sections = (
        None if "all" in {str(s).lower() for s in requested_sections} else requested_sections
    )
    configure_logging(
        profile=str(args.log_profile),
        level=args.log_level,
        log_file=log_path_file,
        enabled_sections=selected_sections,
        min_console_level=args.log_min_level,
        force=True,
    )
    logger = get_section_logger("entry.sim", "main")

    if args.headless:
        # Keep user-provided env vars untouched; only set a conservative default.
        # Avoid forcing EGL/surfaceless globally because some driver stacks stall there.
        os.environ.setdefault("OPEN3D_CPU_RENDERING", "1")

        # Headless means no runtime windows from OmniMap / Fisher visualizer.
        args.vis_gui = False
        args.show_fisher_heatmap = False
        args.show_fisher_arrows = False
        args.step_delay_sec = 0.0
        args.hold_gui_sec = 0.0

        logger.info(
            "[闭环] 已启用无头模式：强制 vis_gui=False、show_fisher_heatmap=False、"
            "show_fisher_arrows=False、step_delay_sec=0、hold_gui_sec=0"
        )

    from sim.motion_policy import FisherMotionPolicy
    from sim.omnimap_runner import (
        OmniMapRunner,
        build_fisher_debug_config_overrides,
    )
    from sim.scene_simulator import SceneSimulator

    # 1. Build the static simulated scene once.
    scene_t0 = time.perf_counter()
    logger.info("[闭环] 阶段 1/4：初始化 SceneSimulator")
    simulator = SceneSimulator(width=args.width, height=args.height)
    logger.info("[闭环] 阶段 1/4：加载点云")
    simulator.load_pointcloud(
        args.pcd_path,
        voxel_size=args.voxel_size,
        scale=args.point_scale,
    )
    simulator.set_intrinsics(args.fx, args.fy, args.cx, args.cy)
    if args.ground:
        simulator.add_ground(size=args.ground_size, z=args.ground_z)
    if args.coord_frame:
        simulator.add_coordinate_frame()
    logger.info("[闭环] 阶段 1/4 完成，耗时 %.2fs", time.perf_counter() - scene_t0)

    stats = simulator.get_scene_stats()
    logger.info("[闭环] 场景统计：")
    for key, value in stats.items():
        logger.info("  - %s: %s", key, value)
    if simulator.aabb is not None:
        extent = np.asarray(simulator.aabb.get_extent(), dtype=np.float64)
        extent_max = float(np.max(np.abs(extent)))
        if args.point_scale >= 1.0 and extent_max > 20.0:
            logger.warning(
                "[闭环] 警告：场景尺度过大。"
                f"max_extent={extent_max:.3f} (world units), point_scale={args.point_scale}. "
                "If this point cloud is in millimeters, use --point_scale 0.001 to avoid extreme depth and TSDF memory spikes."
            )

    if simulator.scene_center is None or simulator.aabb is None:
        raise RuntimeError("scene_center/aabb unavailable after simulator setup")

    center = np.asarray(simulator.scene_center, dtype=np.float64)
    extent = np.asarray(simulator.aabb.get_extent(), dtype=np.float64)
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
    base_radius = (
        float(args.hemi_radius)
        if args.hemi_radius is not None
        else 0.5 * float(np.linalg.norm(extent)) + 0.3
    )
    if args.cartesian:
        displacement_per_step = float(args.linear_vel_max) * float(args.dt)
        radius_ratio = displacement_per_step / max(base_radius, 1e-9)
        if radius_ratio > 0.2:
            logger.warning(
                "[闭环] 警告：笛卡尔线速度步进可能过激，"
                "current reference sphere. "
                f"linear_vel_max*dt={displacement_per_step:.4f}, "
                f"radius={base_radius:.4f}, ratio={radius_ratio:.4f}. "
                "If the object leaves the view frustum, reduce --linear_vel_max or --dt."
            )
    # 2. Initialize the first pose on the upper hemisphere around the scene.
    current_c2w = spherical_c2w(
        center=center,
        radius=base_radius,
        theta=float(args.init_theta),
        phi=float(args.init_phi),
    )
    intrinsics = np.array([args.fx, args.fy, args.cx, args.cy], dtype=np.float32)

    runner = OmniMapRunner.from_config_path(
        # 2. Initialize OmniMap runtime (TSDF + 3DGS + Fisher components).
        config_path=args.config,
        output=str(save_dir),
        depth_scale=args.depth_scale,
        vis_gui=args.vis_gui,
        scene=args.scene,
        max_depth_m=args.max_depth,
        config_overrides=fisher_config_overrides,
        verbose=True,
        log_every=max(1, int(args.log_every)),
    )
    logger.info("[闭环] 阶段 2/4：OmniMapRunner 已就绪")
    policy = FisherMotionPolicy(
        fisher_step_scale=step_scale,
        cartesian=args.cartesian,
        dt=args.dt,
        radial_gain=args.radial_gain,
        linear_vel_max=args.linear_vel_max,
        angular_gain=args.angular_gain,
        angular_speed_max=args.angular_speed_max,
        enable_angular=args.enable_angular,
        grad_eps=args.grad_eps,
        spherical_speed_min=args.spherical_speed_min,
        max_delta_theta=args.max_delta_theta,
        max_delta_phi=args.max_delta_phi,
        verbose=True,
        orientation_roll_mode="current_frame_min_roll",
        control_law_mode="dt_consistent",
        
    )
    logger.info("[闭环] 阶段 3/4：FisherMotionPolicy 已就绪")

    log_path = save_dir / "loop_log.jsonl"
    csv_path = save_dir / "loop_debug.csv"
    with (
        open(log_path, "w", encoding="utf-8") as log_file,
        open(csv_path, "w", encoding="utf-8", newline="") as csv_file,
    ):
        timing_totals = {
            "render": 0.0,
            "track": 0.0,
            "policy": 0.0,
            "io": 0.0,
            "step_total": 0.0,
        }
        timing_count = 0
        csv_writer = csv.DictWriter(
            csv_file,
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
                "delta_theta_applied",
                "delta_phi_applied",
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
                "num_keyframes",
                "num_gaussians",
                "depth_min_m",
                "depth_max_m",
                "should_stop",
                "stop_reason",
            ],
        )
        csv_writer.writeheader()
        last_processed_idx = -1
        for idx in range(int(args.num_steps)):
            last_processed_idx = idx
            step_t0 = time.perf_counter()
            # 3. Render the current pose into RGBD.
            t_render0 = time.perf_counter()
            render_result = simulator.render(current_c2w)
            t_render = time.perf_counter() - t_render0
            rgb = render_result.rgb
            depth = render_result.depth
            depth_min, depth_max, valid_count, total_count, valid_ratio = (
                compute_depth_stats(depth)
            )

            if args.save_frames:
                save_render_artifacts(
                    save_dir=save_dir,
                    idx=idx,
                    rgb=rgb,
                    depth=depth,
                    c2w=current_c2w,
                )

            if valid_count == 0:
                logger.warning(
                    f"[闭环] 在 step={idx} 的 TSDF 融合前停止："
                    "rendered depth has no valid pixels. "
                    f"valid_ratio={valid_ratio:.6f}, "
                    f"cam_pos={current_c2w[:3, 3].tolist()}"
                )
                break

            # 4. Push the rendered frame into OmniMap.
            t_track0 = time.perf_counter()
            step_result = runner.step(
                idx=idx,
                rgb=rgb,
                depth_m=depth,
                c2w=current_c2w,
                intrinsics_vec=intrinsics,
                is_last=bool(args.terminate and idx == int(args.num_steps) - 1),
            )
            t_track = time.perf_counter() - t_track0

            # 5. Query Fisher and derive the next pose for the following step.
            t_policy0 = time.perf_counter()
            motion_result = policy.next_pose_from_c2w(
                gs_backend=runner.omni.gs,
                current_c2w=current_c2w,
                intrinsics_vec=intrinsics,
                image_size=(args.height, args.width),
                idx=idx + 1,
            )
            t_policy = time.perf_counter() - t_policy0

            # 6. Log enough state to audit both mapping progress and control behavior.
            t_io0 = time.perf_counter()
            log_entry = {
                "idx": idx,
                "current_c2w": current_c2w.tolist(),
                "camera_center": current_c2w[:3, 3].tolist(),
                "current_theta": motion_result.current_theta,
                "current_phi": motion_result.current_phi,
                "next_theta": motion_result.next_theta,
                "next_phi": motion_result.next_phi,
                "grad_theta_raw": motion_result.grad_theta_raw,
                "grad_phi_raw": motion_result.grad_phi_raw,
                "grad_theta_compressed": motion_result.grad_theta_compressed,
                "grad_phi_compressed": motion_result.grad_phi_compressed,
                "grad_norm_raw": motion_result.grad_norm_raw,
                "grad_norm_compressed": motion_result.grad_norm_compressed,
                "fisher_current_score": motion_result.fisher_score,
                "scaled_theta": motion_result.scaled_theta,
                "scaled_phi": motion_result.scaled_phi,
                "delta_theta_applied": motion_result.delta_theta_applied,
                "delta_phi_applied": motion_result.delta_phi_applied,
                "fisher_step_scale": float(args.fisher_step_scale),
                "cartesian": bool(args.cartesian),
                "dt": float(args.dt),
                "radial_gain": float(args.radial_gain),
                "linear_vel_max": float(args.linear_vel_max),
                "angular_gain": float(args.angular_gain),
                "enable_angular": bool(args.enable_angular),
                "step_scale_theta": motion_result.step_scale_theta,
                "step_scale_phi": motion_result.step_scale_phi,
                "speed_clipped": motion_result.speed_clipped,
                "clip_scale_ratio": motion_result.clip_scale_ratio,
                "num_keyframes": step_result.num_keyframes,
                "num_gaussians": step_result.num_gaussians,
                "depth_min_m": step_result.depth_min_m,
                "depth_max_m": step_result.depth_max_m,
                "spherical_speed_raw": motion_result.spherical_speed_raw,
                "spherical_speed_scaled": motion_result.spherical_speed_scaled,
                "spherical_speed_applied": motion_result.spherical_speed_applied,
                "spherical_speed_limit": motion_result.spherical_speed_limit,
                "spherical_speed_min": motion_result.spherical_speed_min,
                "should_stop": motion_result.should_stop,
                "stop_reason": motion_result.stop_reason,
                "reference_radius": motion_result.reference_radius,
                "current_radius": motion_result.current_radius,
                "radial_error": motion_result.radial_error,
                "velocity_raw_world": motion_result.velocity_raw_world.tolist(),
                "vt_world": motion_result.vt_world.tolist(),
                "vn_world": motion_result.vn_world.tolist(),
                "velocity_world": motion_result.velocity_world.tolist(),
                "rotvec_error": motion_result.rotvec_error.tolist(),
                "angular_velocity_world": motion_result.angular_velocity_world.tolist(),
                "angular_speed_raw": motion_result.angular_speed_raw,
                "angular_speed_applied": motion_result.angular_speed_applied,
                "angular_gain": motion_result.angular_gain,
                "enable_angular": motion_result.enable_angular,
                "desired_c2w": motion_result.desired_c2w.tolist(),
                "linear_speed_raw": motion_result.linear_speed_raw,
                "linear_speed_applied": motion_result.linear_speed_applied,
                "linear_speed_limit": motion_result.linear_speed_limit,
                "next_position": motion_result.next_position,
                "fisher_visualization": {
                    "show_fisher_heatmap": bool(args.show_fisher_heatmap),
                    "show_fisher_arrows": bool(args.show_fisher_arrows),
                    "fisher_arrow_length": float(args.fisher_arrow_length),
                    "fisher_debug_log": bool(args.fisher_debug_log),
                    "fisher_window_mode": str(args.fisher_window_mode),
                    "fisher_heatmap_window_name": str(args.fisher_heatmap_window_name),
                    "fisher_velocity_window_name": str(
                        args.fisher_velocity_window_name
                    ),
                    "fisher_num_samples": int(args.fisher_num_samples),
                    "fisher_num_dense_points": int(args.fisher_num_dense_points),
                    "fisher_idw_power": float(args.fisher_idw_power),
                    "fisher_display_radius_scale": float(
                        args.fisher_display_radius_scale
                    ),
                    "fisher_arrow_radius_scale": float(args.fisher_arrow_radius_scale),
                },
                "next_c2w": motion_result.next_c2w.tolist(),
            }
            log_file.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
            csv_writer.writerow(
                {
                    "idx": idx,
                    "controller_mode": motion_result.controller_mode,
                    "cartesian": motion_result.cartesian,
                    "fisher_step_scale": float(args.fisher_step_scale),
                    "dt": float(args.dt),
                    "radial_gain": float(args.radial_gain),
                    "linear_vel_max": float(args.linear_vel_max),
                    "angular_gain": float(args.angular_gain),
                    "enable_angular": bool(args.enable_angular),
                    "grad_theta_raw": motion_result.grad_theta_raw,
                    "grad_phi_raw": motion_result.grad_phi_raw,
                    "grad_theta_compressed": motion_result.grad_theta_compressed,
                    "grad_phi_compressed": motion_result.grad_phi_compressed,
                    "scaled_theta": motion_result.scaled_theta,
                    "scaled_phi": motion_result.scaled_phi,
                    "delta_theta_applied": motion_result.delta_theta_applied,
                    "delta_phi_applied": motion_result.delta_phi_applied,
                    "speed_clipped": motion_result.speed_clipped,
                    "clip_scale_ratio": motion_result.clip_scale_ratio,
                    "grad_norm_raw": motion_result.grad_norm_raw,
                    "grad_norm_compressed": motion_result.grad_norm_compressed,
                    "fisher_score": motion_result.fisher_score,
                    "spherical_speed_raw": motion_result.spherical_speed_raw,
                    "spherical_speed_scaled": motion_result.spherical_speed_scaled,
                    "spherical_speed_applied": motion_result.spherical_speed_applied,
                    "spherical_speed_limit": motion_result.spherical_speed_limit,
                    "spherical_speed_min": motion_result.spherical_speed_min,
                    "reference_radius": motion_result.reference_radius,
                    "current_radius": motion_result.current_radius,
                    "radial_error": motion_result.radial_error,
                    "vt_world_norm": float(np.linalg.norm(motion_result.vt_world)),
                    "vn_world_norm": float(np.linalg.norm(motion_result.vn_world)),
                    "velocity_raw_world_norm": float(
                        np.linalg.norm(motion_result.velocity_raw_world)
                    ),
                    "velocity_world_norm": float(
                        np.linalg.norm(motion_result.velocity_world)
                    ),
                    "linear_speed_raw": motion_result.linear_speed_raw,
                    "linear_speed_applied": motion_result.linear_speed_applied,
                    "linear_speed_limit": motion_result.linear_speed_limit,
                    "angular_speed_raw": motion_result.angular_speed_raw,
                    "angular_speed_applied": motion_result.angular_speed_applied,
                    "rotvec_error_norm": float(
                        np.linalg.norm(motion_result.rotvec_error)
                    ),
                    "angular_velocity_world_norm": float(
                        np.linalg.norm(motion_result.angular_velocity_world)
                    ),
                    "max_scale_before_clip": (
                        float("nan")
                        if args.cartesian
                        else float(np.hypot(args.max_delta_theta, args.max_delta_phi))
                        / max(motion_result.grad_norm_raw, 1e-12)
                    ),
                    "num_keyframes": step_result.num_keyframes,
                    "num_gaussians": step_result.num_gaussians,
                    "depth_min_m": step_result.depth_min_m,
                    "depth_max_m": step_result.depth_max_m,
                    "should_stop": motion_result.should_stop,
                    "stop_reason": motion_result.stop_reason,
                }
            )
            force_sync_logs = (
                should_log_step(idx, int(args.log_every))
                or motion_result.should_stop
                or idx == int(args.num_steps) - 1
            )
            if force_sync_logs:
                log_file.flush()
                csv_file.flush()
            t_io = time.perf_counter() - t_io0
            step_total = time.perf_counter() - step_t0
            timing_totals["render"] += t_render
            timing_totals["track"] += t_track
            timing_totals["policy"] += t_policy
            timing_totals["io"] += t_io
            timing_totals["step_total"] += step_total
            timing_count += 1
            timing_msg = (
                f"[计时] step={idx} 渲染={t_render:.3f}s "
                f"跟踪={t_track:.3f}s 策略={t_policy:.3f}s "
                f"日志IO={t_io:.3f}s 总计={step_total:.3f}s"
            )
            if should_log_step(idx, int(args.log_every)):
                logger.info(timing_msg)
            else:
                logger.debug(timing_msg)

            step_message = (
                (
                    f"[闭环] step={idx} 模式=笛卡尔 "
                    f"r={motion_result.current_radius:.4f} "
                    f"dr={motion_result.radial_error:.6f} "
                    f"|vt|={np.linalg.norm(motion_result.vt_world):.6f} "
                    f"|vn|={np.linalg.norm(motion_result.vn_world):.6f} "
                    f"|v_raw|={np.linalg.norm(motion_result.velocity_raw_world):.6f} "
                    f"|v|={np.linalg.norm(motion_result.velocity_world):.6f} "
                    f"|rotvec_err|={motion_result.angular_speed_raw:.6f} "
                    f"|omega|={motion_result.angular_speed_applied:.6f} "
                    f"停止={motion_result.should_stop} "
                    f"关键帧数={step_result.num_keyframes} "
                    f"高斯数={step_result.num_gaussians}"
                )
                if args.cartesian
                else (
                    f"[闭环] step={idx} theta={motion_result.current_theta:.4f} "
                    f"phi={motion_result.current_phi:.4f} -> next_theta={motion_result.next_theta:.4f} "
                    f"next_phi={motion_result.next_phi:.4f} "
                    f"增量=({motion_result.delta_theta_applied:.4f}, {motion_result.delta_phi_applied:.4f}) "
                    f"停止={motion_result.should_stop} "
                    f"关键帧数={step_result.num_keyframes} "
                    f"高斯数={step_result.num_gaussians}"
                )
            )
            if (
                should_log_step(idx, int(args.log_every))
                or motion_result.should_stop
                or idx == int(args.num_steps) - 1
            ):
                logger.info(step_message)
            else:
                logger.debug(step_message)

            if motion_result.should_stop:
                logger.info(
                    f"[闭环] 在 step={idx} 提前停止："
                    f"|u_scaled|={motion_result.spherical_speed_scaled:.6f} "
                    f"< min={motion_result.spherical_speed_min:.6f}"
                )
                break

            current_c2w = motion_result.next_c2w
            if args.step_delay_sec > 0:
                time.sleep(float(args.step_delay_sec))
        if timing_count > 0:
            logger.info(
                "[计时汇总] avg_render=%.3fs avg_track=%.3fs avg_policy=%.3fs avg_io=%.3fs avg_total=%.3fs，共 %d 步",
                timing_totals["render"] / timing_count,
                timing_totals["track"] / timing_count,
                timing_totals["policy"] / timing_count,
                timing_totals["io"] / timing_count,
                timing_totals["step_total"] / timing_count,
                timing_count,
            )

    np.save(save_dir / "trajectory_c2w_last.npy", current_c2w)
    logger.info("[闭环] 已保存逐步日志：%s 和 %s", log_path, csv_path)
    if args.vis_gui:
        runner.export_final_fisher_artifacts(tag="final")
    elif (not args.terminate) and last_processed_idx >= 0:
        runner.export_fisher_snapshot(
            pose_c2w=current_c2w,
            idx=int(last_processed_idx),
            tag="final",
        )
        logger.info(
            "[闭环] 无头模式最终 Fisher 快照已导出，step=%d",
            int(last_processed_idx),
        )

    if args.terminate:
        runner.terminate()
    if args.vis_gui and args.hold_gui_sec > 0:
        logger.info("[闭环] 退出前保持 GUI 显示 %.2fs", args.hold_gui_sec)
        time.sleep(float(args.hold_gui_sec))


def main() -> None:
    """CLI entrypoint for advanced closed-loop experiments."""
    args = parse_args()
    run_closed_loop(args)


if __name__ == "__main__":
    main()
