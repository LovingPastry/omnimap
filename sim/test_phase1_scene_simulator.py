from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from sim.scene_simulator import SceneSimulator


def look_at_c2w(
    eye: np.ndarray,
    target: np.ndarray,
    up: np.ndarray | None = None,
) -> np.ndarray:
    """Build a camera-to-world matrix for a camera looking at ``target``.

    Convention:
    - camera looks along its +Z axis in local frame for this helper
    - returned matrix is `c2w`

    This helper is only for Phase-1 testing. The exact convention only needs to
    be self-consistent so we can verify that rendering responds correctly to pose
    changes before integrating with OmniMap.
    """
    eye = np.asarray(eye, dtype=np.float64).reshape(3)
    target = np.asarray(target, dtype=np.float64).reshape(3)
    up = np.array([0.0, 0.0, 1.0], dtype=np.float64) if up is None else np.asarray(up, dtype=np.float64).reshape(3)

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

    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, 0] = right
    c2w[:3, 1] = true_up
    c2w[:3, 2] = forward
    c2w[:3, 3] = eye
    return c2w


def save_render_result(prefix: Path, rgb: np.ndarray, depth: np.ndarray) -> None:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(prefix.with_suffix(".png")), rgb_bgr)
    np.save(str(prefix.with_name(prefix.name + "_depth.npy")), depth)

    valid = np.isfinite(depth) & (depth > 0)
    if np.any(valid):
        d_min = float(depth[valid].min())
        d_max = float(depth[valid].max())
        denom = max(d_max - d_min, 1e-6)
        norm = np.clip((depth - d_min) / denom, 0.0, 1.0)
    else:
        norm = np.zeros_like(depth, dtype=np.float32)
    vis = (norm * 255.0).astype(np.uint8)
    vis = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
    cv2.imwrite(str(prefix.with_name(prefix.name + "_depth_vis.png")), vis)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase-1 test for sim.scene_simulator.SceneSimulator")
    parser.add_argument("--pointcloud", required=True, help="Path to a .ply or .pcd point cloud")
    parser.add_argument("--output_dir", default="sim_outputs/phase1", help="Directory to save debug renders")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fx", type=float, default=525.0)
    parser.add_argument("--fy", type=float, default=525.0)
    parser.add_argument("--cx", type=float, default=319.5)
    parser.add_argument("--cy", type=float, default=239.5)
    parser.add_argument("--voxel_size", type=float, default=None)
    parser.add_argument("--ground", action="store_true", help="Add a ground plane")
    parser.add_argument("--ground_size", type=float, default=4.0)
    parser.add_argument("--ground_z", type=float, default=0.0)
    parser.add_argument("--coord_frame", action="store_true", help="Add a coordinate frame for debugging")
    parser.add_argument("--radius_scale", type=float, default=1.5, help="Scale factor applied to scene extent to place test cameras")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    simulator = SceneSimulator(width=args.width, height=args.height)
    simulator.load_pointcloud(args.pointcloud, voxel_size=args.voxel_size)
    simulator.set_intrinsics(args.fx, args.fy, args.cx, args.cy)

    if args.ground:
        simulator.add_ground(size=args.ground_size, z=args.ground_z)
    if args.coord_frame:
        simulator.add_coordinate_frame()

    stats = simulator.get_scene_stats()
    print("[SceneSimulator] scene stats:")
    for key, value in stats.items():
        print(f"  - {key}: {value}")

    if simulator.scene_center is None or simulator.aabb is None:
        raise RuntimeError("scene_center/aabb unavailable after point cloud loading")

    center = simulator.scene_center
    extent = np.asarray(simulator.aabb.get_extent(), dtype=np.float64)
    base_radius = max(float(np.linalg.norm(extent)), 1.0) * float(args.radius_scale)

    eyes = [
        center + np.array([base_radius, 0.0, base_radius * 0.35], dtype=np.float64),
        center + np.array([0.0, -base_radius, base_radius * 0.40], dtype=np.float64),
        center + np.array([-0.8 * base_radius, 0.6 * base_radius, base_radius * 0.55], dtype=np.float64),
    ]

    for idx, eye in enumerate(eyes):
        c2w = look_at_c2w(eye=eye, target=center)
        result = simulator.render(c2w)
        prefix = output_dir / f"view_{idx:02d}_rgb"
        save_render_result(prefix, result.rgb, result.depth)
        np.save(str(output_dir / f"view_{idx:02d}_c2w.npy"), c2w)
        valid = np.isfinite(result.depth) & (result.depth > 0)
        depth_min = float(result.depth[valid].min()) if np.any(valid) else float("nan")
        depth_max = float(result.depth[valid].max()) if np.any(valid) else float("nan")
        print(
            f"[Render {idx}] eye={eye.tolist()} depth_min={depth_min:.4f} depth_max={depth_max:.4f} "
            f"saved_prefix={prefix}"
        )

    print(f"Phase-1 test finished. Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
