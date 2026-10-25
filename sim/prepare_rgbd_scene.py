"""Build a coloured simulation point cloud from one measured RGB-D frame.

No object masks or semantic labels are used. Table and background are retained
so target selection must be performed by the ordinary YOLO/TAP/SBERT mapper.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sim.sim_fisher_closed_loop import spherical_c2w


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rgb", default="data/cup/imap/00/rgb/0.png")
    parser.add_argument("--depth", default="data/cup/imap/00/depth/0.png")
    parser.add_argument("--output", default="sim/sim_outputs/semantic_cup_scene/rgbd_scene.ply")
    parser.add_argument("--fx", type=float, default=525.)
    parser.add_argument("--fy", type=float, default=525.)
    parser.add_argument("--cx", type=float, default=319.5)
    parser.add_argument("--cy", type=float, default=239.5)
    parser.add_argument("--depth_scale", type=float, default=1000.)
    parser.add_argument("--max_depth", type=float, default=.75)
    parser.add_argument("--voxel_size", type=float, default=.0015)
    args = parser.parse_args()
    rgb = cv2.imread(args.rgb)
    depth = cv2.imread(args.depth, cv2.IMREAD_UNCHANGED)
    if rgb is None or depth is None:
        raise FileNotFoundError("RGB-D source frame not found")
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    if rgb.shape[:2] != depth.shape:
        raise ValueError("RGB and depth sizes differ")
    z = depth.astype(np.float64) / args.depth_scale
    v, u = np.mgrid[:z.shape[0], :z.shape[1]]
    xyz = np.stack(((u - args.cx) * z / args.fx, (v - args.cy) * z / args.fy, z), axis=-1)
    keep = np.isfinite(z) & (z > .1) & (z < args.max_depth)
    transform = spherical_c2w(center=np.zeros(3), radius=.5, theta=0, phi=.6)
    points = xyz[keep] @ transform[:3, :3].T + transform[:3, 3]
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.colors = o3d.utility.Vector3dVector(rgb[keep] / 255.)
    cloud = cloud.voxel_down_sample(args.voxel_size)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_point_cloud(str(output), cloud):
        raise RuntimeError(f"could not save {output}")
    metadata = {
        **vars(args), "num_points": len(cloud.points), "source_camera_c2w": transform.tolist(),
        "note": "Single-view measured RGB-D geometry; no object masks or ground-truth ROI.",
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata))


if __name__ == "__main__":
    main()
