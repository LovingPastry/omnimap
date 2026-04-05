from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from sim.omnimap_runner import OmniMapRunner


def parse_args() -> argparse.Namespace:
    """Parse the Phase-2 runner smoke-test CLI options."""
    parser = argparse.ArgumentParser(
        description=(
            "Phase-2 test for the simulation pipeline: load Phase-1 RGBD outputs "
            "from disk and feed them directly into OmniMap without ROS or topics."
        ),
        epilog=(
            "Example:\n"
            "  python3 sim/test_phase2_omnimap_runner.py \\\n"
            "    --input_dir sim/sim_outputs/phase1 \\\n"
            "    --config config/sim_rtabmap_config.yaml \\\n"
            "    --output sim/sim_outputs/phase2 \\\n"
            "    --scene room_0 \\\n"
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
        help="Path to OmniMap yaml config",
    )
    parser.add_argument(
        "--output",
        default="sim/sim_outputs/phase2",
        help="Directory to store OmniMap outputs",
    )
    parser.add_argument("--fx", type=float, default=525.0)
    parser.add_argument("--fy", type=float, default=525.0)
    parser.add_argument("--cx", type=float, default=319.5)
    parser.add_argument("--cy", type=float, default=239.5)
    parser.add_argument("--depth_scale", type=float, default=1000.0)
    parser.add_argument("--max_depth", type=float, default=None)
    parser.add_argument(
        "--scene",
        type=str,
        default="room_0",
        help="Scene name passed through to OmniMap GUI branches",
    )
    parser.add_argument("--vis_gui", action="store_true")
    parser.add_argument(
        "--terminate",
        action="store_true",
        help="Call omni.terminate() after feeding all frames",
    )
    return parser.parse_args()


def load_phase1_triplets(input_dir: Path) -> list[tuple[int, Path, Path, Path]]:
    """Collect the RGB/depth/c2w triplets emitted by the Phase-1 script."""
    triplets: list[tuple[int, Path, Path, Path]] = []
    for rgb_path in sorted(input_dir.glob("view_*_rgb.png")):
        stem = rgb_path.stem.replace("_rgb", "")
        idx = int(stem.split("_")[1])
        depth_path = input_dir / f"{stem}_rgb_depth.npy"
        c2w_path = input_dir / f"{stem}_c2w.npy"
        if not depth_path.exists():
            raise FileNotFoundError(
                f"Missing depth file for {rgb_path.name}: {depth_path}"
            )
        if not c2w_path.exists():
            raise FileNotFoundError(f"Missing c2w file for {rgb_path.name}: {c2w_path}")
        triplets.append((idx, rgb_path, depth_path, c2w_path))
    if not triplets:
        raise FileNotFoundError(f"No Phase-1 view_*_rgb.png files found in {input_dir}")
    return triplets


def load_tsdf_depth_max(config_path: str) -> float:
    """Read the TSDF depth limit from the OmniMap yaml config."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return float(cfg.get("tsdf", {}).get("depth_max", 20.0))


def validate_phase1_inputs(
    triplets: list[tuple[int, Path, Path, Path]],
    *,
    tsdf_depth_max_m: float,
) -> None:
    """Fail fast when a Phase-1 export is clearly stale or scale-mismatched."""
    depth_mins: list[float] = []
    depth_maxs: list[float] = []
    depth_medians: list[float] = []
    cam_dists: list[float] = []

    for _, _, depth_path, c2w_path in triplets:
        depth_m = np.load(depth_path).astype(np.float32)
        valid = np.isfinite(depth_m) & (depth_m > 0)
        if not np.any(valid):
            raise RuntimeError(f"No valid positive depth values found in {depth_path}")
        depth_valid = depth_m[valid]
        depth_mins.append(float(depth_valid.min()))
        depth_maxs.append(float(depth_valid.max()))
        depth_medians.append(float(np.median(depth_valid)))

        c2w = np.load(c2w_path).astype(np.float64)
        cam_dists.append(float(np.linalg.norm(c2w[:3, 3])))

    overall_depth_max = max(depth_maxs)
    overall_depth_median = float(np.median(np.asarray(depth_medians, dtype=np.float64)))
    max_cam_dist = max(cam_dists)
    print(
        f"[Preflight] depth_min={min(depth_mins):.4f} depth_median={overall_depth_median:.4f} "
        f"depth_max={overall_depth_max:.4f} tsdf_depth_max={tsdf_depth_max_m:.4f} "
        f"max_cam_dist={max_cam_dist:.4f}"
    )

    if overall_depth_median > 20.0 or max_cam_dist > 50.0:
        print(
            "[Preflight] Warning: the rendered scene appears extremely large-scale "
            "(for example mm-units interpreted as meters). If this is an object-scale "
            "scene, rerun Phase 1 with a point-scale such as `--point_scale 0.001`."
        )

    if overall_depth_max > tsdf_depth_max_m:
        raise RuntimeError(
            "Phase-1 inputs exceed the TSDF integration depth range. "
            f"Observed depth_max={overall_depth_max:.4f} m, but config tsdf.depth_max={tsdf_depth_max_m:.4f} m. "
            "These Phase-1 outputs are likely from a much larger-scale or stale scene. "
            "Either rerun Phase-1 into a clean output directory, or increase tsdf.depth_max in the config intentionally."
        )


def main() -> None:
    """Replay saved Phase-1 RGBD frames directly into OmniMap."""
    args = parse_args()
    input_dir = Path(args.input_dir)
    intrinsics = np.array([args.fx, args.fy, args.cx, args.cy], dtype=np.float32)
    triplets = load_phase1_triplets(input_dir)
    tsdf_depth_max_m = load_tsdf_depth_max(args.config)
    validate_phase1_inputs(triplets, tsdf_depth_max_m=tsdf_depth_max_m)

    runner = OmniMapRunner.from_config_path(
        config_path=args.config,
        output=args.output,
        depth_scale=args.depth_scale,
        vis_gui=args.vis_gui,
        scene=args.scene,
        max_depth_m=args.max_depth,
        verbose=True,
    )

    # Replay each saved frame in-order so this script mirrors the intended Phase-2 data flow.
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

    if args.terminate:
        runner.terminate()


if __name__ == "__main__":
    main()
