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

    常用参数（仅保留以下 8 项）:
    1) --pcd_path
    2) --save_dir
    3) --radial_gain
    4) --angular_gain
    5) --grad_eps
    6) --dt
    7) --linear_vel_max
    8) --angular_speed_max
    """
    parser = argparse.ArgumentParser(
        description=(
            "Simplified sim entrypoint with only two modes:\n"
            "- default: headless (no GUI)\n"
            "- --vis_gui: split Fisher heatmap/velocity windows\n"
            "Controller is fixed to Cartesian + angular control."
        ),
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
    return parser.parse_args()


def build_advanced_args(simple_args: argparse.Namespace) -> argparse.Namespace:
    """Map simplified arguments to the advanced closed-loop runtime."""
    args = parse_advanced_args([])

    # Exposed parameters
    args.pcd_path = str(simple_args.pcd_path)
    args.save_dir = str(simple_args.save_dir)
    args.radial_gain = float(simple_args.radial_gain)
    args.angular_gain = float(simple_args.angular_gain)
    args.grad_eps = float(simple_args.grad_eps)
    args.dt = float(simple_args.dt)
    args.linear_vel_max = float(simple_args.linear_vel_max)
    args.angular_speed_max = float(simple_args.angular_speed_max)
    args.vis_gui = bool(simple_args.vis_gui)

    # Fixed policy + visualization defaults for simplified mode.
    args.cartesian = True
    args.enable_angular = True
    args.show_fisher_heatmap = True
    args.show_fisher_arrows = True
    args.fisher_window_mode = "split"

    # Normalize no-GUI mode without exposing an extra headless flag.
    args.headless = not bool(simple_args.vis_gui)
    args.headless_dense_fisher_export = not bool(simple_args.vis_gui)

    return args


def main() -> None:
    simple_args = parse_args()
    advanced_args = build_advanced_args(simple_args)
    run_closed_loop(advanced_args)


if __name__ == "__main__":
    main()
