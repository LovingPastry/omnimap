from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Keep script-mode imports stable: python3 sim/main.py
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from sim.sim_fisher_closed_loop import parse_args as parse_advanced_args
from sim.sim_fisher_closed_loop import run_closed_loop


def parse_args() -> argparse.Namespace:
    """Simplified daily-entry CLI for simulation closed-loop runs.

    常用参数（仅保留以下 9 项）:
    1) --pcd_path
    2) --save_dir
    3) --fisher_step_scale
    4) --radial_gain
    5) --angular_gain
    6) --grad_eps
    7) --dt
    8) --linear_vel_max
    9) --angular_speed_max
    """
    parser = argparse.ArgumentParser(
        description=(
            "Simplified sim entrypoint with only two modes:\n"
            "- default: headless (no GUI)\n"
            "- --vis_gui: split Fisher heatmap/velocity windows\n"
            "Controller is fixed to Cartesian + angular control."
        ),
        epilog="""
        python3 sim/main.py \
        --pcd_path replica/log_pcd_0/对比结果/RoboSeg_geo/Kettle.ply \
        --save_dir sim/sim_outputs/phase4_main_gui \
        --fisher_step_scale 1e-4 \
        --radial_gain 0.2 \
        --angular_gain 2.0 \
        --grad_eps 0.01 \
        --dt 0.1 \
        --linear_vel_max 0.5 \
        --angular_speed_max 1.0 \
        --vis_gui
        """,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--pcd_path",
        required=True,
        help="Path to a .ply/.pcd point cloud",
    )
    parser.add_argument(
        "--save_dir",
        default="sim/sim_outputs/phase4_main",
        help="Output directory for logs and artifacts",
    )
    parser.add_argument(
        "--fisher_step_scale",
        type=float,
        default=1e-4,
        help="Primary Fisher control scale applied after log-compressed gradient normalization",
    )
    parser.add_argument(
        "--radial_gain",
        type=float,
        default=0.2,
        help="Radial correction gain",
    )
    parser.add_argument(
        "--angular_gain",
        type=float,
        default=2.0,
        help="Angular gain for orientation correction",
    )
    parser.add_argument(
        "--grad_eps",
        type=float,
        default=0.01,
        help="Finite-difference epsilon for Fisher gradient query",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.1,
        help="Control integration timestep (seconds)",
    )
    parser.add_argument(
        "--linear_vel_max",
        type=float,
        default=0.5,
        help="Maximum Cartesian linear speed",
    )
    parser.add_argument(
        "--angular_speed_max",
        type=float,
        default=1.0,
        help="Maximum angular speed magnitude for omega clipping (rad/s)",
    )
    parser.add_argument(
        "--vis_gui",
        action="store_true",
        help="Enable split GUI windows for Fisher heatmap and velocity field",
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
    return parser.parse_args()


def build_advanced_args(simple_args: argparse.Namespace) -> argparse.Namespace:
    """Map simplified arguments to the advanced closed-loop runtime."""
    # Seed required args for the advanced parser first, then override below.
    # parse_advanced_args requires --pcd_path at parse time.
    args = parse_advanced_args(["--pcd_path", str(simple_args.pcd_path)])

    # Exposed parameters
    args.pcd_path = str(simple_args.pcd_path)
    args.save_dir = str(simple_args.save_dir)
    args.fisher_step_scale = float(simple_args.fisher_step_scale)
    args.radial_gain = float(simple_args.radial_gain)
    args.angular_gain = float(simple_args.angular_gain)
    args.grad_eps = float(simple_args.grad_eps)
    args.dt = float(simple_args.dt)
    args.linear_vel_max = float(simple_args.linear_vel_max)
    args.angular_speed_max = float(simple_args.angular_speed_max)
    args.vis_gui = bool(simple_args.vis_gui)
    args.log_profile = str(simple_args.log_profile)
    args.log_level = simple_args.log_level
    args.log_every = int(simple_args.log_every)
    args.log_file = bool(simple_args.log_file)

    # Fixed policy + visualization defaults for simplified mode.
    args.cartesian = True
    args.enable_angular = True
    args.show_fisher_heatmap = True
    args.show_fisher_arrows = True
    args.fisher_window_mode = "split"

    # Normalize no-GUI mode without exposing an extra headless flag.
    args.headless = not bool(simple_args.vis_gui)

    return args


def main() -> None:
    simple_args = parse_args()
    advanced_args = build_advanced_args(simple_args)
    run_closed_loop(advanced_args)


if __name__ == "__main__":
    main()
