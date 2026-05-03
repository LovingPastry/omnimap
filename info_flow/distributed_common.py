import os  # nopep8

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import sys  # nopep8
from pathlib import Path  # nopep8

REPO_ROOT = Path(__file__).resolve().parent.parent
OMNIMAP_ROOT = REPO_ROOT / "omnimap"
for path in (REPO_ROOT, OMNIMAP_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import os
import resource
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import rospy
import tf2_ros
import torch
from geometry_msgs.msg import PoseStamped, TransformStamped, TwistStamped
from omnimap.util.utils import configure_logging, load_config
from planner_snapshot import (
    PlannerSnapshot,
    load_planner_snapshot_file,
    save_planner_snapshot_file,
)
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import CameraInfo


def set_nofile_limit() -> None:
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (100000, rlimit[1]))


def configure_entry_logging(args) -> None:
    os.makedirs(args.output, exist_ok=True)
    log_file_path = os.path.join(args.output, "run.log") if bool(args.log_file) else None
    requested_sections = args.log_section or ["all"]
    selected_sections = (
        None
        if "all" in {str(section).lower() for section in requested_sections}
        else requested_sections
    )
    configure_logging(
        profile=str(args.log_profile),
        level=args.log_level,
        log_file=log_file_path,
        enabled_sections=selected_sections,
        min_console_level=getattr(args, "log_min_level", "INFO"),
        force=True,
    )


def load_runtime_config(config_path: str) -> dict:
    config = load_config(config_path)
    config.setdefault("fisher_num_samples", 128)
    config.setdefault("fisher_num_dense_points", 1024)
    mc = config.setdefault("motion_control", {})
    mc.setdefault("linear_vel_max", 0.05)
    mc.setdefault("angular_speed_max", 1.0)
    mc.setdefault("enable_angular", True)
    mc.setdefault("planner_hz", 30.0)
    mc.setdefault("pose_stale_timeout_sec", 0.2)
    mc.setdefault("fisher_step_scale", 1e-5)
    mc.setdefault("angular_gain", 2.0)
    mc.setdefault("radial_gain", 0.2)
    mc.setdefault("dt", 1.0)
    mc.setdefault("grad_eps", 0.01)
    mc.setdefault("spherical_speed_min", 0.0)
    mc.setdefault("servo_hz", 50.0)
    mc.setdefault("spherical_cmd_timeout_sec", 0.25)
    return config


def import_omnimap_msgs():
    try:
        from omnimap_msgs.msg import PlannerSnapshotRef, SphericalCommand
    except Exception as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError(
            "failed to import omnimap_msgs; run `catkin_make` under ros_ws and source source_env.sh first"
        ) from exc
    return PlannerSnapshotRef, SphericalCommand


def wait_for_camera_calibration(topic: str):
    cam_info_msg = rospy.wait_for_message(topic, CameraInfo)
    K = np.array(cam_info_msg.K, dtype=np.float64).reshape(3, 3)
    calib = np.array(
        [K[0, 0], K[1, 1], K[0, 2], K[1, 2]],
        dtype=np.float32,
    )
    image_size_hw = (int(cam_info_msg.height), int(cam_info_msg.width))
    return cam_info_msg, K, calib, image_size_hw


def c2w_to_w2c_posevec(c2w: np.ndarray) -> np.ndarray:
    w2c = np.linalg.inv(np.asarray(c2w, dtype=np.float64))
    quat = R.from_matrix(w2c[:3, :3]).as_quat()
    return np.hstack((w2c[:3, 3], quat)).astype(np.float64)


def transform_to_c2w(transform: TransformStamped) -> np.ndarray:
    t = transform.transform.translation
    q = transform.transform.rotation
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, 3] = [t.x, t.y, t.z]
    c2w[:3, :3] = R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
    return c2w


def lookup_pose_from_tf(
    tf_buffer: tf2_ros.Buffer,
    *,
    world_frame: str,
    camera_frame: str,
    stamp: rospy.Time,
    timeout_sec: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray, rospy.Time]:
    transform: TransformStamped = tf_buffer.lookup_transform(
        world_frame,
        camera_frame,
        stamp,
        rospy.Duration(timeout_sec),
    )
    c2w = transform_to_c2w(transform)
    return c2w_to_w2c_posevec(c2w), c2w, transform.header.stamp


def lookup_latest_pose_from_tf(
    tf_buffer: tf2_ros.Buffer,
    *,
    world_frame: str,
    camera_frame: str,
    timeout_sec: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray, rospy.Time]:
    transform: TransformStamped = tf_buffer.lookup_transform(
        world_frame,
        camera_frame,
        rospy.Time(0),
        rospy.Duration(timeout_sec),
    )
    c2w = transform_to_c2w(transform)
    return c2w_to_w2c_posevec(c2w), c2w, transform.header.stamp


def pose_stamped_from_c2w(
    *,
    c2w: np.ndarray,
    stamp: rospy.Time,
    frame_id: str,
) -> PoseStamped:
    c2w = np.asarray(c2w, dtype=np.float64)
    quat = R.from_matrix(c2w[:3, :3]).as_quat()
    msg = PoseStamped()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.pose.position.x = float(c2w[0, 3])
    msg.pose.position.y = float(c2w[1, 3])
    msg.pose.position.z = float(c2w[2, 3])
    msg.pose.orientation.x = float(quat[0])
    msg.pose.orientation.y = float(quat[1])
    msg.pose.orientation.z = float(quat[2])
    msg.pose.orientation.w = float(quat[3])
    return msg


def c2w_from_pose_stamped(msg: PoseStamped) -> np.ndarray:
    quat = np.array(
        [
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w,
        ],
        dtype=np.float64,
    )
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = R.from_quat(quat).as_matrix()
    c2w[:3, 3] = np.array(
        [
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
        ],
        dtype=np.float64,
    )
    return c2w


def publish_zero_twist(pub, *, cmd_frame: str, stamp: Optional[rospy.Time] = None) -> None:
    msg = TwistStamped()
    msg.header.stamp = stamp if stamp is not None else rospy.Time.now()
    msg.header.frame_id = cmd_frame
    pub.publish(msg)


def publish_motion_components(
    pub,
    *,
    cmd_frame: str,
    linear_cmd: np.ndarray,
    angular_cmd: np.ndarray,
    stamp: Optional[rospy.Time] = None,
) -> None:
    msg = TwistStamped()
    msg.header.stamp = stamp if stamp is not None else rospy.Time.now()
    msg.header.frame_id = cmd_frame
    linear = np.asarray(linear_cmd, dtype=np.float64).reshape(3)
    angular = np.asarray(angular_cmd, dtype=np.float64).reshape(3)
    msg.twist.linear.x = float(linear[0])
    msg.twist.linear.y = float(linear[1])
    msg.twist.linear.z = float(linear[2])
    msg.twist.angular.x = float(angular[0])
    msg.twist.angular.y = float(angular[1])
    msg.twist.angular.z = float(angular[2])
    pub.publish(msg)


def position_to_spherical(
    position: np.ndarray,
    scene_center: np.ndarray,
) -> Tuple[float, float, float]:
    offset = np.asarray(position, dtype=np.float64).reshape(3) - np.asarray(
        scene_center,
        dtype=np.float64,
    ).reshape(3)
    radius = float(np.linalg.norm(offset))
    if radius < 1e-12:
        raise ValueError("position is too close to scene center")
    n_hat = offset / radius
    theta = float(np.arctan2(n_hat[1], n_hat[0]) % (2.0 * np.pi))
    phi = float(np.arcsin(np.clip(n_hat[2], 0.0, 1.0)))
    return radius, theta, phi


def local_frame_from_theta_phi(theta: float, phi: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ct, st = float(np.cos(theta)), float(np.sin(theta))
    cp, sp = float(np.cos(phi)), float(np.sin(phi))
    e_theta = np.array([-cp * st, cp * ct, 0.0], dtype=np.float64)
    e_phi = np.array([-sp * ct, -sp * st, cp], dtype=np.float64)
    n_hat = np.array([cp * ct, cp * st, sp], dtype=np.float64)
    return e_theta, e_phi, n_hat


def look_at_c2w(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    eye = np.asarray(eye, dtype=np.float64).reshape(3)
    target = np.asarray(target, dtype=np.float64).reshape(3)
    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    forward = target - eye
    forward_norm = np.linalg.norm(forward)
    if forward_norm < 1e-12:
        raise ValueError("eye and target are too close")
    forward = forward / forward_norm

    right = np.cross(forward, up)
    right_norm = np.linalg.norm(right)
    if right_norm < 1e-12:
        fallback_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        right = np.cross(forward, fallback_up)
        right_norm = np.linalg.norm(right)
        if right_norm < 1e-12:
            raise ValueError("failed to construct right axis")
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


def fixed_hemisphere_from_config(config: dict) -> Tuple[Optional[np.ndarray], float]:
    center = None
    radius = 0.35
    tsdf_cfg = config.get("tsdf", {}) if isinstance(config, dict) else {}
    bounds = tsdf_cfg.get("spatial_bounds", None) if isinstance(tsdf_cfg, dict) else None
    if isinstance(bounds, (list, tuple)) and len(bounds) == 6:
        x_min, x_max, y_min, y_max, z_min, z_max = [float(v) for v in bounds]
        center = np.array(
            [
                0.5 * (x_min + x_max),
                0.5 * (y_min + y_max),
                0.5 * (z_min + z_max),
            ],
            dtype=np.float64,
        )
    return center, float(radius)


def apply_fixed_hemisphere_reference(motion_policy, center: Optional[np.ndarray], radius: float) -> None:
    if center is None:
        return
    center_np = np.asarray(center, dtype=np.float64).reshape(3)
    motion_policy.reference_scene_center = center_np.copy()
    motion_policy.reference_radius = float(radius)
    motion_policy.reference_initialized = True


def make_run_id(prefix: str = "omnimap") -> str:
    return f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"


@dataclass
class CachedPose:
    stamp: rospy.Time
    pose_w2c: np.ndarray
    pose_4x4: np.ndarray
    wall_time: float


@dataclass
class CachedSnapshotRef:
    run_id: str
    model_version: int
    keyframe_idx: int
    snapshot_uri: str
    created_wall_time: float
    runtime_device_hint: str
    stamp: rospy.Time
    receipt_wall_time: float


@dataclass
class CachedSphericalCommand:
    msg: object
    receipt_wall_time: float


class SnapshotStore:
    def __init__(
        self,
        *,
        root_dir: str,
        run_id: str,
        retention: int = 4,
        logger=None,
    ) -> None:
        self.root_dir = Path(root_dir).expanduser().resolve()
        self.run_id = str(run_id)
        self.retention = max(1, int(retention))
        self.run_dir = self.root_dir / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger

    def snapshot_path(self, model_version: int) -> Path:
        return self.run_dir / f"snapshot_v{int(model_version):06d}.pt"

    def save(self, snapshot: PlannerSnapshot) -> Path:
        path = self.snapshot_path(snapshot.model_version)
        tmp_path = path.with_suffix(".pt.tmp")
        save_planner_snapshot_file(snapshot, tmp_path)
        with open(tmp_path, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
        self._prune()
        return path

    def load(self, snapshot_uri: str) -> PlannerSnapshot:
        return load_planner_snapshot_file(snapshot_uri)

    def _prune(self) -> None:
        snapshot_paths = sorted(self.run_dir.glob("snapshot_v*.pt"))
        excess = len(snapshot_paths) - self.retention
        if excess <= 0:
            return
        for stale_path in snapshot_paths[:excess]:
            try:
                stale_path.unlink()
            except FileNotFoundError:
                continue
            except OSError as exc:
                if self.logger is not None:
                    self.logger.warning("删除旧快照失败：%s (%s)", str(stale_path), exc)
