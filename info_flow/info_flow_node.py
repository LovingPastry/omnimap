import os  # nopep8

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys  # nopep8
from pathlib import Path  # nopep8

REPO_ROOT = Path(__file__).resolve().parent.parent
OMNIMAP_ROOT = REPO_ROOT / "omnimap"
for path in (REPO_ROOT, OMNIMAP_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import argparse
import resource
import threading
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
import rospy
import tf2_ros
import torch
from geometry_msgs.msg import TransformStamped, TwistStamped
from message_filters import ApproximateTimeSynchronizer, Subscriber
from omni import OMNI
from omnimap.gaussian.renderer.nbv.motion_policy import FisherMotionPolicy
from omnimap.util.utils import (
    configure_logging,
    get_section_logger,
    load_config,
    should_log_step,
)
from planner_snapshot import PlannerSnapshot, build_planner_snapshot
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import CameraInfo, CompressedImage
from slam_frontend import DropOldestQueue, RTabMapKeyframeGate
from tqdm import tqdm, trange

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (100000, rlimit[1]))


@dataclass
class TrackTask:
    index: int
    stamp: rospy.Time
    pose_w2c: np.ndarray
    pose_4x4: np.ndarray
    image: torch.Tensor
    depth: torch.Tensor
    image_shape: Tuple[int, int]
    depth_valid_ratio: float
    source: str


@dataclass
class PoseState:
    stamp: rospy.Time
    pose_w2c: np.ndarray
    pose_4x4: np.ndarray
    wall_time: float


@dataclass
class SphericalCommand:
    stamp: rospy.Time
    wall_time: float
    model_version: int
    idx: int
    dt: float
    theta_rate: float
    phi_rate: float
    reference_radius: float
    reference_scene_center: np.ndarray
    should_stop: bool
    stop_reason: str
    fisher_score: float


@dataclass
class PipelineStats:
    pose_updates: int = 0
    frame_candidates: int = 0
    frames_enqueued: int = 0
    queue_dropped: int = 0
    track_success: int = 0
    track_fail: int = 0
    snapshot_success: int = 0
    snapshot_fail: int = 0
    planner_steps: int = 0
    planner_nonzero: int = 0
    planner_zero: int = 0
    planner_zero_missing_input: int = 0
    planner_zero_pose_stale: int = 0
    planner_zero_policy_stop: int = 0
    planner_zero_exception: int = 0
    servo_steps: int = 0
    servo_nonzero: int = 0
    servo_zero: int = 0
    servo_zero_missing_cmd: int = 0
    servo_zero_pose_stale: int = 0
    servo_zero_cmd_stale: int = 0
    gated_by_interval: int = 0
    gated_by_motion: int = 0
    gated_passed: int = 0
    forced_gap_keyframes: int = 0


def save_trajectory(omni, all_inputs, output):
    """
    保存轨迹和图像为 Replica 数据集标准格式。
    """
    base_path = os.path.join(output, "imap", "00")
    rgb_out = os.path.join(base_path, "rgb")
    depth_out = os.path.join(base_path, "depth")
    os.makedirs(rgb_out, exist_ok=True)
    os.makedirs(depth_out, exist_ok=True)

    np.save(f"{output}/intrinsics.npy", omni.intrinsics.cpu().numpy())

    traj_full = []
    for i in trange(len(all_inputs), desc="Saving frames", unit="frame"):
        frame = all_inputs[i]

        rgb_image = frame["image"][0]
        rgb_image = rgb_image.transpose(1, 2, 0)
        rgb_image = cv2.cvtColor(rgb_image.astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(rgb_out, f"{i}.png"), rgb_image)

        depth_image = frame["depth"][0]
        depth_scale = frame["depth_scale"]
        depth_image = (depth_image * depth_scale).astype(np.uint16)
        cv2.imwrite(os.path.join(depth_out, f"{i}.png"), depth_image)

        pose_44 = frame["pose_44"][0]
        traj_full.append(pose_44.flatten())

    traj_full = np.stack(traj_full)
    np.savetxt(
        os.path.join(base_path, "traj_w_c.txt"),
        traj_full,
        fmt="%.18e",
        delimiter=" ",
    )

    logger = get_section_logger("entry.infoflow", "main")
    logger.info("已保存 %d 帧到 %s", len(traj_full), base_path)
    logger.info("  - RGB 图像：%s", rgb_out)
    logger.info("  - 深度图像：%s", depth_out)
    logger.info("  - 轨迹文件：%s", f"{base_path}/traj_w_c.txt")


class InfoFlowROSNode:
    def __init__(self, args, config):
        self.main_logger = get_section_logger("entry.infoflow", "main")
        self.planner_logger = get_section_logger("planner.infoflow", "planner")
        self.profile_logger = get_section_logger("profile.infoflow", "profile")
        self.logger = self.main_logger
        rospy.init_node("info_flow_node", anonymous=True)
        self._throttle_last = {}

        self.main_logger.info("正在等待相机内参消息...")
        cam_info_msg = rospy.wait_for_message(args.camera_info_topic, CameraInfo)
        self.K = np.array(cam_info_msg.K, dtype=np.float64).reshape(3, 3)
        self.depth_scale = float(args.depth_scale)
        self.calib = np.array(
            [self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]],
            dtype=np.float32,
        )
        self.intrinsics = torch.tensor(self.calib.copy())
        self.image_size_hw = (int(cam_info_msg.height), int(cam_info_msg.width))

        self.log_every = max(1, int(args.log_every))
        self.progress_bar = tqdm(
            desc="InfoFlow",
            disable=(str(args.log_profile).lower() == "quiet"),
        )
        self.max_frames = int(args.max_frames)
        if self.max_frames <= 0:
            raise ValueError(f"max_frames must be positive, got {self.max_frames}")

        self.terminate_enabled = bool(args.terminate)
        self.save_fisher_snapshots = bool(args.save_fisher_snapshots)
        self.shutdown_requested = False
        self.first_snapshot_saved = False
        self.last_pose_4x4 = None

        self.output = args.output
        self.should_record_frames = self.terminate_enabled and self.output != "None"
        self.can_export_fisher_snapshots = self.save_fisher_snapshots and self.output != "None"
        self.all_inputs = []
        if self.should_record_frames or self.can_export_fisher_snapshots:
            os.makedirs(self.output, exist_ok=True)

        self.cmd_frame = args.cmd_frame
        self.cmd_pub = rospy.Publisher(args.cmd_topic, TwistStamped, queue_size=1)

        requested_mode = str(args.slam_frontend_mode)
        self.mode = "tf_native" if requested_mode == "rtabmap_native" else requested_mode
        self.slam_odom_topic = str(args.slam_odom_topic)
        self.odom_info_topic = str(args.odom_info_topic)
        self.planner_hz = float(args.planner_hz)
        self.servo_hz = float(args.servo_hz)
        self.pose_stale_timeout_sec = float(args.pose_stale_timeout_sec)
        self.spherical_cmd_timeout_sec = float(args.spherical_cmd_timeout_sec)
        self.planner_output_mode = str(args.planner_output_mode)
        if self.planner_hz <= 0 or self.servo_hz <= 0:
            raise ValueError(
                f"planner_hz and servo_hz must be positive, got planner_hz={self.planner_hz}, servo_hz={self.servo_hz}"
            )
        if self.spherical_cmd_timeout_sec <= 0:
            raise ValueError(
                f"spherical_cmd_timeout_sec must be positive, got {self.spherical_cmd_timeout_sec}"
            )
        self.sync_slop_sec = float(args.sync_slop_sec)
        self.start_wall_time = float(time.monotonic())
        self.status_log_interval_sec = max(0.2, float(args.status_log_interval_sec))
        self.force_keyframe_gap_frames = 30  # fixed by design
        self._last_status_log_wall = self.start_wall_time
        self._last_status_planner_steps = 0
        self._last_status_planner_nonzero = 0
        self._last_status_planner_zero = 0
        self._last_status_servo_steps = 0
        self._last_status_servo_nonzero = 0
        self._last_status_servo_zero = 0
        self._last_queue_drop_log_wall = 0.0
        self._tf_diag_prev_pose_4x4 = None
        self._tf_diag_prev_stamp_sec = None
        self._last_sync_cb_wall = 0.0
        self._last_sync_stamp_sec = None
        self._last_tf_success_wall = 0.0
        self._last_tf_success_stamp_sec = None

        self.pose_lock = threading.Lock()
        self.latest_pose_state: Optional[PoseState] = None

        self.snapshot_lock = threading.Lock()
        self.active_snapshot: Optional[PlannerSnapshot] = None
        self.model_version = 0
        self.servo_lock = threading.Lock()
        self.active_spherical_cmd: Optional[SphericalCommand] = None

        self.stats_lock = threading.Lock()
        self.stats = PipelineStats()

        self.next_track_index = 0
        self.planner_tick = 0

        self.omni = OMNI(args, config)
        self.motion_policy = FisherMotionPolicy(
            fisher_step_scale=args.fisher_step_scale,
            cartesian=True,
            dt=args.dt,
            radial_gain=args.radial_gain,
            linear_vel_max=args.linear_vel_max,
            angular_gain=args.angular_gain,
            angular_speed_max=args.angular_speed_max,
            enable_angular=args.enable_angular,
            grad_eps=args.grad_eps,
            spherical_speed_min=args.spherical_speed_min,
            planner_output_mode=self.planner_output_mode,
            verbose=True,
            # control_law_mode="dt_consistent",
        )
        self.motion_policy_lock = threading.Lock()
        self.min_valid_reference_radius_m = 0.3
        self.fixed_hemisphere_center: Optional[np.ndarray] = None
        self.fixed_hemisphere_radius_m: float = 0.35

        tsdf_cfg = config.get("tsdf", {}) if isinstance(config, dict) else {}
        bounds = tsdf_cfg.get("spatial_bounds", None) if isinstance(tsdf_cfg, dict) else None
        if isinstance(bounds, (list, tuple)) and len(bounds) == 6:
            x_min, x_max, y_min, y_max, z_min, z_max = [float(v) for v in bounds]
            self.fixed_hemisphere_center = np.array(
                [
                    0.5 * (x_min + x_max),
                    0.5 * (y_min + y_max),
                    0.5 * (z_min + z_max),
                ],
                dtype=np.float64,
            )
            self.main_logger.warning(
                (
                    "临时策略启用：半球中心固定为 spatial_bounds 中心=%s，"
                    "参考半径固定为 %.3fm"
                ),
                self.fixed_hemisphere_center.tolist(),
                float(self.fixed_hemisphere_radius_m),
            )
            self._apply_fixed_hemisphere_reference()

        self.rgb_sub = None
        self.depth_sub = None
        self.tsync = None
        self.keyframe_gate = None
        self.tf_buffer = None
        self.tf_listener = None
        self.world_frame = args.world_frame
        self.camera_frame = args.camera_frame

        self.track_queue = None
        self.track_stop_event = None
        self.track_worker_thread = None
        self.planner_timer = None
        self.servo_timer = None
        self.spherical_cmd_pub = rospy.Publisher(
            "/omnimap/spherical_cmd", TwistStamped, queue_size=1
        )

        if requested_mode == "rtabmap_native":
            self.main_logger.info("slam_frontend_mode=rtabmap_native 已迁移为 tf_native（兼容别名）。")

        if self.mode == "tf_native":
            self.keyframe_gate = RTabMapKeyframeGate(
                min_interval_sec=float(args.keyframe_min_interval_sec),
                min_translation_m=float(args.keyframe_min_translation_m),
                min_rotation_deg=float(args.keyframe_min_rotation_deg),
                forced_gap_frames=int(self.force_keyframe_gap_frames),
            )

        if self.mode == "tf_native":
            self._setup_tf_native(args)
        elif self.mode == "legacy_tf":
            self._setup_legacy_tf(args)
        else:
            raise ValueError(f"Unknown slam_frontend_mode={requested_mode!r}; expected tf_native, rtabmap_native(alias), or legacy_tf")

        self.main_logger.info(
            "InfoFlowROSNode 初始化完成，模式=%s，TwistStamped 发布到 %s",
            self.mode,
            args.cmd_topic,
        )
        self.main_logger.info(
            "控制输出模式：planner_output_mode=%s planner_hz=%.1f servo_hz=%.1f",
            self.planner_output_mode,
            float(self.planner_hz),
            float(self.servo_hz),
        )
        self.main_logger.info(
            "前端位姿参数默认值：slam_frontend_mode=%s world_frame=%s camera_frame=%s",
            self.mode,
            self.world_frame,
            self.camera_frame,
        )

    def _setup_tf_native(self, args):
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.track_queue = DropOldestQueue(maxsize=int(args.track_queue_size))
        self.track_stop_event = threading.Event()
        self.track_worker_thread = threading.Thread(
            target=self._tracking_worker,
            name="infoflow_tracking_worker",
            daemon=True,
        )
        self.track_worker_thread.start()

        self.rgb_sub = Subscriber(args.rgb_topic, CompressedImage)
        self.depth_sub = Subscriber(args.depth_topic, CompressedImage)
        self.tsync = ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub],
            queue_size=30,
            slop=max(0.0, float(args.sync_slop_sec)),
        )
        self.tsync.registerCallback(self.tf_native_callback)

        self.planner_timer = rospy.Timer(
            rospy.Duration(1.0 / max(self.planner_hz, 1e-6)),
            self._planning_timer_callback,
        )
        if self.planner_output_mode == "spherical_rate":
            self.servo_timer = rospy.Timer(
                rospy.Duration(1.0 / max(self.servo_hz, 1e-6)),
                self._servo_timer_callback,
            )

    def _setup_legacy_tf(self, args):
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.rgb_sub = Subscriber(args.rgb_topic, CompressedImage)
        self.depth_sub = Subscriber(args.depth_topic, CompressedImage)
        self.tsync = ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub],
            queue_size=10,
            slop=0.1,
        )
        self.tsync.registerCallback(self.legacy_image_callback)

    @staticmethod
    def _c2w_to_w2c_posevec(c2w: np.ndarray) -> np.ndarray:
        w2c = np.linalg.inv(np.asarray(c2w, dtype=np.float64))
        quat = R.from_matrix(w2c[:3, :3]).as_quat()
        return np.hstack((w2c[:3, 3], quat)).astype(np.float64)

    @staticmethod
    def _pose_delta_metrics(prev_c2w: np.ndarray, curr_c2w: np.ndarray) -> Tuple[float, float]:
        prev = np.asarray(prev_c2w, dtype=np.float64)
        curr = np.asarray(curr_c2w, dtype=np.float64)
        trans_m = float(np.linalg.norm(curr[:3, 3] - prev[:3, 3]))
        rel = prev[:3, :3].T @ curr[:3, :3]
        cos_theta = max(-1.0, min(1.0, float((np.trace(rel) - 1.0) * 0.5)))
        rot_deg = float(np.degrees(np.arccos(cos_theta)))
        return trans_m, rot_deg

    def _set_latest_pose(self, stamp: rospy.Time, pose_4x4: np.ndarray, pose_w2c: np.ndarray):
        state = PoseState(
            stamp=stamp,
            pose_w2c=np.asarray(pose_w2c, dtype=np.float64),
            pose_4x4=np.asarray(pose_4x4, dtype=np.float64),
            wall_time=float(time.monotonic()),
        )
        with self.pose_lock:
            self.latest_pose_state = state
        with self.stats_lock:
            self.stats.pose_updates += 1

    def _decode_compressed_rgbd(self, rgb_msg, depth_msg):
        rgb_buffer = np.frombuffer(rgb_msg.data, dtype=np.uint8)
        rgb_bgr = cv2.imdecode(rgb_buffer, cv2.IMREAD_COLOR)
        if rgb_bgr is None:
            raise ValueError("failed to decode compressed RGB image")
        rgb_image = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)

        depth_buffer = np.frombuffer(depth_msg.data, dtype=np.uint8)
        depth_image = cv2.imdecode(depth_buffer, cv2.IMREAD_UNCHANGED)
        if depth_image is None:
            raise ValueError("failed to decode compressed depth image")
        if depth_image.ndim == 3:
            depth_image = cv2.cvtColor(depth_image, cv2.COLOR_BGR2GRAY)

        return rgb_image, depth_image

    def _build_track_task(
        self,
        *,
        idx: int,
        stamp: rospy.Time,
        pose_w2c: np.ndarray,
        pose_4x4: np.ndarray,
        rgb_image: np.ndarray,
        depth_image: np.ndarray,
        source: str,
    ) -> TrackTask:
        image = torch.as_tensor(rgb_image).permute(2, 0, 1)
        depth = torch.as_tensor(depth_image.astype(np.float32) / self.depth_scale)
        return TrackTask(
            index=int(idx),
            stamp=stamp,
            pose_w2c=np.asarray(pose_w2c, dtype=np.float64),
            pose_4x4=np.asarray(pose_4x4, dtype=np.float64),
            image=image,
            depth=depth,
            image_shape=(int(rgb_image.shape[0]), int(rgb_image.shape[1])),
            depth_valid_ratio=float(np.mean(depth_image > 0)),
            source=str(source),
        )

    def _enqueue_track_task(self, task: TrackTask):
        dropped = self.track_queue.put(task)
        with self.stats_lock:
            self.stats.frames_enqueued += 1
            if dropped is not None:
                self.stats.queue_dropped += 1
                dropped_total = int(self.stats.queue_dropped)
            else:
                dropped_total = int(self.stats.queue_dropped)
        if dropped is not None:
            now = float(time.monotonic())
            if (now - self._last_queue_drop_log_wall) >= self.status_log_interval_sec:
                self._last_queue_drop_log_wall = now
                self.main_logger.debug(
                    "跟踪队列积压：丢弃最旧帧 idx=%d，新帧 idx=%d，队列=%d，累计丢弃=%d",
                    int(dropped.index),
                    int(task.index),
                    int(self.track_queue.qsize()),
                    dropped_total,
                )

    def _log_throttle(self, key: str, interval_sec: float, level: str, msg: str, *args):
        now = float(time.monotonic())
        last = float(self._throttle_last.get(key, -1e18))
        if (now - last) < float(interval_sec):
            return
        self._throttle_last[key] = now
        getattr(self.main_logger, str(level).lower())(msg, *args)

    def _planner_log_throttle(self, key: str, interval_sec: float, level: str, msg: str, *args):
        now = float(time.monotonic())
        last = float(self._throttle_last.get(f"planner:{key}", -1e18))
        if (now - last) < float(interval_sec):
            return
        self._throttle_last[f"planner:{key}"] = now
        getattr(self.planner_logger, str(level).lower())(msg, *args)

    def _maybe_log_planning_status(self):
        now = float(time.monotonic())
        elapsed = now - self._last_status_log_wall
        if elapsed < self.status_log_interval_sec:
            return

        with self.stats_lock:
            steps = int(self.stats.planner_steps)
            nonzero = int(self.stats.planner_nonzero)
            zero = int(self.stats.planner_zero)
            servo_steps = int(self.stats.servo_steps)
            servo_nonzero = int(self.stats.servo_nonzero)
            servo_zero = int(self.stats.servo_zero)
            servo_zero_missing_cmd = int(self.stats.servo_zero_missing_cmd)
            servo_zero_pose_stale = int(self.stats.servo_zero_pose_stale)
            servo_zero_cmd_stale = int(self.stats.servo_zero_cmd_stale)
            zero_missing = int(self.stats.planner_zero_missing_input)
            zero_stale = int(self.stats.planner_zero_pose_stale)
            zero_stop = int(self.stats.planner_zero_policy_stop)
            zero_exc = int(self.stats.planner_zero_exception)
            track_ok = int(self.stats.track_success)
            dropped = int(self.stats.queue_dropped)
            gate_interval = int(self.stats.gated_by_interval)
            gate_motion = int(self.stats.gated_by_motion)
            gate_passed = int(self.stats.gated_passed)
            gate_forced = int(self.stats.forced_gap_keyframes)

        d_steps = steps - self._last_status_planner_steps
        d_nonzero = nonzero - self._last_status_planner_nonzero
        d_zero = zero - self._last_status_planner_zero
        d_servo_steps = servo_steps - self._last_status_servo_steps
        d_servo_nonzero = servo_nonzero - self._last_status_servo_nonzero
        d_servo_zero = servo_zero - self._last_status_servo_zero
        plan_hz = d_steps / max(elapsed, 1e-6)
        cmd_hz = d_nonzero / max(elapsed, 1e-6)
        zero_hz = d_zero / max(elapsed, 1e-6)
        servo_hz = d_servo_steps / max(elapsed, 1e-6)
        servo_cmd_hz = d_servo_nonzero / max(elapsed, 1e-6)
        servo_zero_hz = d_servo_zero / max(elapsed, 1e-6)

        self._last_status_log_wall = now
        self._last_status_planner_steps = steps
        self._last_status_planner_nonzero = nonzero
        self._last_status_planner_zero = zero
        self._last_status_servo_steps = servo_steps
        self._last_status_servo_nonzero = servo_nonzero
        self._last_status_servo_zero = servo_zero

        self.planner_logger.info(
            (
                "规划状态：model_v=%d plan_hz=%.1f cmd_hz=%.1f zero_hz=%.1f "
                "servo_hz=%.1f servo_cmd_hz=%.1f servo_zero_hz=%.1f "
                "累计(非零=%d 零速=%d) 零速原因(输入缺失=%d 位姿过期=%d 策略停止=%d 异常=%d) "
                "servo累计(非零=%d 零速=%d) servo零速原因(命令缺失=%d 命令过期=%d 位姿过期=%d) "
                "门控(通过=%d interval=%d motion=%d forced=%d) "
                "跟踪成功=%d 队列=%d 丢弃=%d"
            ),
            int(self.model_version),
            float(plan_hz),
            float(cmd_hz),
            float(zero_hz),
            float(servo_hz),
            float(servo_cmd_hz),
            float(servo_zero_hz),
            int(nonzero),
            int(zero),
            int(zero_missing),
            int(zero_stale),
            int(zero_stop),
            int(zero_exc),
            int(servo_nonzero),
            int(servo_zero),
            int(servo_zero_missing_cmd),
            int(servo_zero_cmd_stale),
            int(servo_zero_pose_stale),
            int(gate_passed),
            int(gate_interval),
            int(gate_motion),
            int(gate_forced),
            int(track_ok),
            int(self.track_queue.qsize()) if self.track_queue is not None else -1,
            int(dropped),
        )

    def _tracked_frame_limit_reached(self) -> bool:
        with self.stats_lock:
            return int(self.stats.track_success) >= int(self.max_frames)

    def tf_native_callback(self, rgb_msg, depth_msg):
        if self.shutdown_requested:
            return

        stamp = rgb_msg.header.stamp
        stamp_sec = float(stamp.to_sec()) if stamp is not None else 0.0
        now_sec = float(rospy.Time.now().to_sec())
        cb_wall = float(time.monotonic())
        cb_inter = cb_wall - self._last_sync_cb_wall if self._last_sync_cb_wall > 0.0 else -1.0
        self._last_sync_cb_wall = cb_wall
        self._last_sync_stamp_sec = float(stamp_sec)
        self._log_throttle(
            "tf_native_sync_cb",
            1.0,
            "info",
            "RGBD同步回调触发：rgb_stamp=%.3f now=%.3f lag=%.3fs",
            float(stamp_sec),
            float(now_sec),
            float(now_sec - stamp_sec),
        )
        self.profile_logger.debug(
            "sync_cb: stamp=%.3f ros_lag=%.3fs inter_cb=%.3fs",
            float(stamp_sec),
            float(now_sec - stamp_sec),
            float(cb_inter),
        )
        tf_t0 = time.perf_counter()
        pose_w2c, pose_4x4 = self.get_pose_from_tf(stamp)
        tf_ms = (time.perf_counter() - tf_t0) * 1000.0
        self.profile_logger.debug(
            "tf_lookup: stamp=%.3f tf_ms=%.2f success=%s",
            float(stamp_sec),
            float(tf_ms),
            bool(pose_w2c is not None),
        )
        if pose_w2c is None:
            self._log_throttle(
                "tf_native_sync_tf_fail",
                1.0,
                "warning",
                "RGBD同步已到达，但该帧TF查询失败：rgb_stamp=%.3f now=%.3f lag=%.3fs",
                float(stamp_sec),
                float(now_sec),
                float(now_sec - stamp_sec),
            )
            self.publish_zero_twist(stamp)
            return

        self._set_latest_pose(stamp, pose_4x4, pose_w2c)
        self._last_tf_success_wall = float(time.monotonic())
        self._last_tf_success_stamp_sec = float(stamp_sec)
        self._log_throttle(
            "tf_native_sync_tf_ok",
            1.0,
            "info",
            "TF按图像时间戳查询成功：rgb_stamp=%.3f now=%.3f lag=%.3fs",
            float(stamp_sec),
            float(now_sec),
            float(now_sec - stamp_sec),
        )
        if self._tf_diag_prev_pose_4x4 is not None and self._tf_diag_prev_stamp_sec is not None:
            trans_m, rot_deg = self._pose_delta_metrics(self._tf_diag_prev_pose_4x4, pose_4x4)
            dt_sec = max(0.0, stamp_sec - float(self._tf_diag_prev_stamp_sec))
            self._log_throttle(
                "tf_pose_delta_diag",
                1.0,
                "info",
                "TF位姿增量诊断：dt=%.3fs trans=%.4fm rot=%.3fdeg",
                float(dt_sec),
                float(trans_m),
                float(rot_deg),
            )
        self._tf_diag_prev_pose_4x4 = np.asarray(pose_4x4, dtype=np.float64)
        self._tf_diag_prev_stamp_sec = float(stamp_sec)

        with self.stats_lock:
            self.stats.frame_candidates += 1

        if self._tracked_frame_limit_reached():
            self.request_shutdown(f"达到跟踪帧上限（{self.max_frames}），停止 ROS spin。")
            return

        decision = self.keyframe_gate.decide(
            pose_4x4=pose_4x4,
            stamp_sec=stamp_sec,
            frame_index=int(self.next_track_index),
        )
        if decision.reason == "forced_gap_keyframe":
            with self.stats_lock:
                self.stats.forced_gap_keyframes += 1
            self.main_logger.info(
                "关键帧门控连续拒绝达 %d 帧，已强制当前帧作为 keyframe 入队。",
                int(self.force_keyframe_gap_frames),
            )

        if not decision.should_track:
            with self.stats_lock:
                if decision.reason == "below_min_interval":
                    self.stats.gated_by_interval += 1
                elif decision.reason == "below_motion_threshold":
                    self.stats.gated_by_motion += 1
            self._log_throttle(
                f"gate_reject:{decision.reason}",
                self.status_log_interval_sec,
                "debug",
                "关键帧门控拒绝：reason=%s dt=%.3f trans=%.4f rot=%.3fdeg",
                decision.reason,
                float(decision.dt_sec),
                float(decision.translation_m),
                float(decision.rotation_deg),
            )
            return

        with self.stats_lock:
            self.stats.gated_passed += 1

        try:
            rgb_image, depth_image = self._decode_compressed_rgbd(rgb_msg, depth_msg)
        except Exception as exc:
            self.main_logger.error("压缩图像解码失败：%s", exc)
            self.publish_zero_twist(stamp)
            return

        task = self._build_track_task(
            idx=self.next_track_index,
            stamp=stamp,
            pose_w2c=pose_w2c,
            pose_4x4=pose_4x4,
            rgb_image=rgb_image,
            depth_image=depth_image,
            source=f"tf_keyframe:{decision.reason}",
        )
        self.next_track_index += 1
        self._enqueue_track_task(task)

        if should_log_step(task.index, self.log_every):
            self.main_logger.debug(
                "已入队帧：frame=%d source=%s queue=%d",
                int(task.index),
                task.source,
                int(self.track_queue.qsize()),
            )

    def _tracking_worker(self):
        while not self.track_stop_event.is_set() or not self.track_queue.empty():
            try:
                task = self.track_queue.get(timeout=0.1)
            except TimeoutError:
                continue

            pose_tensor = torch.as_tensor(task.pose_w2c)
            pose_4x4_tensor = torch.as_tensor(task.pose_4x4)

            if self.should_record_frames:
                self.all_inputs.append(
                    {
                        "index": task.index,
                        "image": task.image[None].cpu().numpy(),
                        "depth": task.depth[None].cpu().numpy(),
                        "pose": pose_tensor[None].cpu().numpy(),
                        "intrinsics": self.intrinsics[None].cpu().numpy(),
                        "pose_44": pose_4x4_tensor[None].cpu().numpy(),
                        "is_last": False,
                        "depth_scale": self.depth_scale,
                    }
                )

            try:
                self.omni.track(
                    task.index,
                    task.image[None],
                    task.depth[None],
                    pose_tensor[None],
                    self.progress_bar,
                    intrinsics=self.intrinsics[None],
                    is_last=False,
                    pose_44=pose_4x4_tensor[None],
                    update_rate=self.log_every,
                )
            except Exception as exc:
                self.main_logger.error("跟踪阶段 OMNI.track 执行失败：%s", exc)
                self.publish_zero_twist(task.stamp)
                with self.stats_lock:
                    self.stats.track_fail += 1
                continue

            self.last_pose_4x4 = np.asarray(task.pose_4x4, dtype=np.float64)
            if not self.first_snapshot_saved:
                self.first_snapshot_saved = self.maybe_export_fisher_snapshot(
                    pose_4x4=task.pose_4x4,
                    idx=task.index,
                    tag="first",
                )

            with self.snapshot_lock:
                next_version = self.model_version + 1
            try:
                snapshot = build_planner_snapshot(
                    live_backend=self.omni.gs,
                    model_version=next_version,
                    keyframe_idx=task.index,
                )
                with self.snapshot_lock:
                    self.model_version = snapshot.model_version
                    self.active_snapshot = snapshot
                with self.stats_lock:
                    self.stats.snapshot_success += 1
            except Exception as exc:
                self.planner_logger.error("构建规划快照失败：%s", exc)
                with self.stats_lock:
                    self.stats.snapshot_fail += 1

            with self.stats_lock:
                self.stats.track_success += 1
                tracked_ok = int(self.stats.track_success)

            if should_log_step(task.index, self.log_every):
                with self.stats_lock:
                    dropped = int(self.stats.queue_dropped)
                self.main_logger.info(
                    "跟踪状态：frame=%d source=%s depth_valid_ratio=%.3f model_v=%d queue=%d dropped=%d",
                    int(task.index),
                    task.source,
                    float(task.depth_valid_ratio),
                    int(self.model_version),
                    int(self.track_queue.qsize()) if self.track_queue is not None else -1,
                    int(dropped),
                )

            if tracked_ok >= self.max_frames:
                self.request_shutdown(f"达到跟踪帧上限（{self.max_frames}），停止 ROS spin。")

    def _planning_timer_callback(self, _event):
        planning_t0 = time.perf_counter()
        if self.shutdown_requested:
            return

        with self.pose_lock:
            pose_state = self.latest_pose_state
        with self.snapshot_lock:
            snapshot = self.active_snapshot

        self.planner_tick += 1
        with self.stats_lock:
            self.stats.planner_steps += 1

        if pose_state is None or snapshot is None:
            if self.mode == "tf_native" and pose_state is None and (time.monotonic() - self.start_wall_time) > 2.0:
                self._planner_log_throttle(
                    "pose_wait",
                    5.0,
                    "warning",
                    "尚未从 TF 接收到位姿；请检查 %s -> %s 变换是否可用。",
                    self.world_frame,
                    self.camera_frame,
                )
            self._planner_log_throttle(
                "zero_missing_input",
                2.0,
                "warning",
                "规划发布零速度：pose_state=%s snapshot=%s track_success=%d enqueued=%d queue=%d",
                bool(pose_state is not None),
                bool(snapshot is not None),
                int(self.stats.track_success),
                int(self.stats.frames_enqueued),
                int(self.track_queue.qsize()) if self.track_queue is not None else -1,
            )
            self.publish_zero_twist()
            planning_total_ms = (time.perf_counter() - planning_t0) * 1000.0
            self.profile_logger.info(
                "planning耗时：idx=%d total=%.2fms status=missing_input",
                int(self.planner_tick),
                float(planning_total_ms),
            )
            with self.stats_lock:
                self.stats.planner_zero += 1
                self.stats.planner_zero_missing_input += 1
            self._maybe_log_planning_status()
            return

        pose_age = float((rospy.Time.now() - pose_state.stamp).to_sec())
        if pose_age > self.pose_stale_timeout_sec:
            now_wall = float(time.monotonic())
            since_sync = float(now_wall - self._last_sync_cb_wall) if self._last_sync_cb_wall > 0.0 else -1.0
            since_tf = float(now_wall - self._last_tf_success_wall) if self._last_tf_success_wall > 0.0 else -1.0
            self._planner_log_throttle(
                "pose_stale",
                2.0,
                "warning",
                (
                    "位姿已过期（age=%.3fs > %.3fs），发布零速度。"
                    "最近同步回调距今=%.3fs，最近TF成功距今=%.3fs"
                ),
                pose_age,
                self.pose_stale_timeout_sec,
                since_sync,
                since_tf,
            )
            self.profile_logger.info(
                "pose_stale(planner): pose_age=%.3fs since_sync_cb=%.3fs since_tf_ok=%.3fs stamp=%.3f",
                float(pose_age),
                float(since_sync),
                float(since_tf),
                float(pose_state.stamp.to_sec()),
            )
            self.publish_zero_twist(pose_state.stamp)
            planning_total_ms = (time.perf_counter() - planning_t0) * 1000.0
            self.profile_logger.info(
                "planning耗时：idx=%d total=%.2fms status=pose_stale",
                int(self.planner_tick),
                float(planning_total_ms),
            )
            with self.stats_lock:
                self.stats.planner_zero += 1
                self.stats.planner_zero_pose_stale += 1
            self._maybe_log_planning_status()
            return

        self._maybe_recover_reference_radius(
            snapshot=snapshot,
            pose_state=pose_state,
        )
        self._apply_fixed_hemisphere_reference()

        try:
            with self.motion_policy_lock:
                motion_result = self.motion_policy.next_pose_from_c2w(
                    gs_backend=snapshot.backend,
                    current_c2w=np.asarray(pose_state.pose_4x4, dtype=np.float64),
                    intrinsics_vec=self.calib,
                    image_size=self.image_size_hw,
                    idx=self.planner_tick,
                )
        except Exception as exc:
            self._planner_log_throttle(
                "policy_exception",
                2.0,
                "warning",
                "Fisher 策略规划异常，发布零速度：%s",
                exc,
            )
            self.publish_zero_twist(pose_state.stamp)
            planning_total_ms = (time.perf_counter() - planning_t0) * 1000.0
            self.profile_logger.info(
                "planning耗时：idx=%d total=%.2fms status=policy_exception",
                int(self.planner_tick),
                float(planning_total_ms),
            )
            with self.stats_lock:
                self.stats.planner_zero += 1
                self.stats.planner_zero_exception += 1
            self._maybe_log_planning_status()
            return

        if motion_result.should_stop:
            self._planner_log_throttle(
                "policy_stop",
                2.0,
                "warning",
                ("策略触发停止：reason=%s score=%.6f u_scaled=%.6e u_min=%.6e grad_raw=%.6e grad_comp=%.6e"),
                str(getattr(motion_result, "stop_reason", "unknown")),
                float(getattr(motion_result, "fisher_score", 0.0)),
                float(getattr(motion_result, "spherical_speed_scaled", 0.0)),
                float(getattr(motion_result, "spherical_speed_min", 0.0)),
                float(getattr(motion_result, "spherical_speed_raw", 0.0)),
                float(getattr(motion_result, "grad_norm_compressed", 0.0)),
            )
            self.publish_zero_twist(pose_state.stamp)
            planning_total_ms = (time.perf_counter() - planning_t0) * 1000.0
            self.profile_logger.info(
                "planning耗时：idx=%d total=%.2fms status=policy_stop",
                int(self.planner_tick),
                float(planning_total_ms),
            )
            with self.stats_lock:
                self.stats.planner_zero += 1
                self.stats.planner_zero_policy_stop += 1
            if self.planner_output_mode == "spherical_rate":
                self._cache_spherical_command(
                    motion_result=motion_result,
                    stamp=pose_state.stamp,
                    model_version=int(snapshot.model_version),
                )
            self._maybe_log_planning_status()
            return

        publish_ms = 0.0
        if self.planner_output_mode == "spherical_rate":
            self._cache_spherical_command(
                motion_result=motion_result,
                stamp=pose_state.stamp,
                model_version=int(snapshot.model_version),
            )
            self._publish_spherical_diag(
                motion_result=motion_result,
                stamp=pose_state.stamp,
            )
        else:
            publish_t0 = time.perf_counter()
            self.publish_motion_result(motion_result, pose_state.stamp)
            publish_ms = (time.perf_counter() - publish_t0) * 1000.0
        planning_total_ms = (time.perf_counter() - planning_t0) * 1000.0
        mp_timing = getattr(self.motion_policy, "last_timing", {}) or {}
        fisher_ms = float(mp_timing.get("fisher_ms", float("nan")))
        s2c_ms = float(mp_timing.get("s2c_ms", float("nan")))
        policy_ms = float(mp_timing.get("policy_total_ms", float("nan")))
        self.profile_logger.info(
            ("planning耗时：idx=%d total=%.2fms policy=%.2fms fisher=%.2fms s2c=%.2fms publish=%.2fms model_v=%d"),
            int(self.planner_tick),
            float(planning_total_ms),
            float(policy_ms),
            float(fisher_ms),
            float(s2c_ms),
            float(publish_ms),
            int(self.model_version),
        )
        self._planner_log_throttle(
            "planner_cmd_diag",
            1.0,
            "debug",
            "planner_cmd: mode=%s idx=%d theta_phi_cmd=(%.6f, %.6f) spherical_cmd_age_ms=0.0 cartesian_cmd_norm=%.6f",
            self.planner_output_mode,
            int(self.planner_tick),
            float(getattr(motion_result, "theta_rate_applied", 0.0)),
            float(getattr(motion_result, "phi_rate_applied", 0.0)),
            float(np.linalg.norm(np.asarray(getattr(motion_result, "velocity_world", np.zeros(3)), dtype=np.float64))),
        )
        with self.stats_lock:
            self.stats.planner_nonzero += 1
        self._maybe_log_planning_status()

    def _cache_spherical_command(
        self,
        *,
        motion_result,
        stamp: rospy.Time,
        model_version: int,
    ) -> None:
        cmd = SphericalCommand(
            stamp=stamp,
            wall_time=float(time.monotonic()),
            model_version=int(model_version),
            idx=int(getattr(motion_result, "idx", self.planner_tick)),
            dt=float(getattr(motion_result, "dt", self.motion_policy.dt)),
            theta_rate=float(getattr(motion_result, "theta_rate_applied", 0.0)),
            phi_rate=float(getattr(motion_result, "phi_rate_applied", 0.0)),
            reference_radius=float(getattr(motion_result, "reference_radius", 0.0)),
            reference_scene_center=np.asarray(
                getattr(motion_result, "reference_scene_center", [0.0, 0.0, 0.0]),
                dtype=np.float64,
            ).reshape(3),
            should_stop=bool(getattr(motion_result, "should_stop", False)),
            stop_reason=str(getattr(motion_result, "stop_reason", "unknown")),
            fisher_score=float(getattr(motion_result, "fisher_score", 0.0)),
        )
        with self.servo_lock:
            self.active_spherical_cmd = cmd

    def _publish_spherical_diag(self, *, motion_result, stamp: rospy.Time) -> None:
        msg = TwistStamped()
        msg.header.stamp = stamp if stamp is not None else rospy.Time.now()
        msg.header.frame_id = self.cmd_frame
        msg.twist.angular.x = float(getattr(motion_result, "theta_rate_applied", 0.0))
        msg.twist.angular.y = float(getattr(motion_result, "phi_rate_applied", 0.0))
        msg.twist.angular.z = float(getattr(motion_result, "dt", self.motion_policy.dt))
        msg.twist.linear.x = float(getattr(motion_result, "reference_radius", 0.0))
        msg.twist.linear.y = float(getattr(motion_result, "fisher_score", 0.0))
        self.spherical_cmd_pub.publish(msg)

    @staticmethod
    def _position_to_spherical(position: np.ndarray, scene_center: np.ndarray) -> Tuple[float, float, float]:
        offset = np.asarray(position, dtype=np.float64).reshape(3) - np.asarray(
            scene_center, dtype=np.float64
        ).reshape(3)
        radius = float(np.linalg.norm(offset))
        if radius < 1e-12:
            raise ValueError("position is too close to scene center")
        n_hat = offset / radius
        theta = float(np.arctan2(n_hat[1], n_hat[0]) % (2.0 * np.pi))
        phi = float(np.arcsin(np.clip(n_hat[2], 0.0, 1.0)))
        return radius, theta, phi

    @staticmethod
    def _local_frame_from_theta_phi(theta: float, phi: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        ct, st = float(np.cos(theta)), float(np.sin(theta))
        cp, sp = float(np.cos(phi)), float(np.sin(phi))
        e_theta = np.array([-cp * st, cp * ct, 0.0], dtype=np.float64)
        e_phi = np.array([-sp * ct, -sp * st, cp], dtype=np.float64)
        n_hat = np.array([cp * ct, cp * st, sp], dtype=np.float64)
        return e_theta, e_phi, n_hat

    @staticmethod
    def _look_at_c2w(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
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

    def _servo_timer_callback(self, _event) -> None:
        if self.shutdown_requested:
            return
        with self.stats_lock:
            self.stats.servo_steps += 1

        with self.pose_lock:
            pose_state = self.latest_pose_state
        if pose_state is None:
            self.publish_zero_twist()
            with self.stats_lock:
                self.stats.servo_zero += 1
                self.stats.servo_zero_missing_cmd += 1
            return
        pose_age = float((rospy.Time.now() - pose_state.stamp).to_sec())
        if pose_age > self.pose_stale_timeout_sec:
            now_wall = float(time.monotonic())
            since_sync = float(now_wall - self._last_sync_cb_wall) if self._last_sync_cb_wall > 0.0 else -1.0
            since_tf = float(now_wall - self._last_tf_success_wall) if self._last_tf_success_wall > 0.0 else -1.0
            self._planner_log_throttle(
                "servo_pose_stale",
                1.0,
                "warning",
                "servo 位姿过期（age=%.3fs > %.3fs），发布零速度。",
                pose_age,
                self.pose_stale_timeout_sec,
            )
            self.profile_logger.info(
                "pose_stale(servo): pose_age=%.3fs since_sync_cb=%.3fs since_tf_ok=%.3fs stamp=%.3f",
                float(pose_age),
                float(since_sync),
                float(since_tf),
                float(pose_state.stamp.to_sec()),
            )
            self.publish_zero_twist(pose_state.stamp)
            with self.stats_lock:
                self.stats.servo_zero += 1
                self.stats.servo_zero_pose_stale += 1
            return

        with self.servo_lock:
            cmd = self.active_spherical_cmd
        if cmd is None:
            self.publish_zero_twist(pose_state.stamp)
            with self.stats_lock:
                self.stats.servo_zero += 1
                self.stats.servo_zero_missing_cmd += 1
            return
        cmd_age = float(time.monotonic() - cmd.wall_time)
        if cmd_age > self.spherical_cmd_timeout_sec:
            self._planner_log_throttle(
                "servo_cmd_stale",
                1.0,
                "warning",
                "servo 命令过期（age=%.3fs > %.3fs），发布零速度。",
                cmd_age,
                self.spherical_cmd_timeout_sec,
            )
            self.publish_zero_twist(pose_state.stamp)
            with self.stats_lock:
                self.stats.servo_zero += 1
                self.stats.servo_zero_cmd_stale += 1
            return
        if cmd.should_stop:
            self.publish_zero_twist(pose_state.stamp)
            with self.stats_lock:
                self.stats.servo_zero += 1
            return

        try:
            current_c2w = np.asarray(pose_state.pose_4x4, dtype=np.float64)
            current_position = current_c2w[:3, 3]
            reference_scene_center = np.asarray(
                cmd.reference_scene_center, dtype=np.float64
            ).reshape(3)
            radius, theta, phi = self._position_to_spherical(
                current_position,
                reference_scene_center,
            )
            e_theta, e_phi, n_hat = self._local_frame_from_theta_phi(theta, phi)
            theta_rate = float(cmd.theta_rate)
            phi_rate = float(cmd.phi_rate)
            v_t = radius * (theta_rate * e_theta + phi_rate * e_phi)
            radial_error = float(cmd.reference_radius - radius)
            if abs(radial_error) > float(self.motion_policy.radial_deadband):
                v_r = float(self.motion_policy.radial_gain) * radial_error * n_hat
            else:
                v_r = np.zeros(3, dtype=np.float64)
            linear_cmd = v_t + v_r
            linear_norm_raw = float(np.linalg.norm(linear_cmd))
            linear_scale = 1.0
            if linear_norm_raw > float(self.motion_policy.linear_vel_max):
                linear_scale = float(self.motion_policy.linear_vel_max) / max(
                    linear_norm_raw, 1e-12
                )
                linear_cmd = linear_cmd * linear_scale

            desired_c2w = self._look_at_c2w(current_position, reference_scene_center)
            current_rotation = np.asarray(current_c2w[:3, :3], dtype=np.float64)
            desired_rotation = np.asarray(desired_c2w[:3, :3], dtype=np.float64)
            rotation_error = desired_rotation @ current_rotation.T
            rotvec_error = R.from_matrix(rotation_error).as_rotvec().astype(np.float64)
            angular_cmd = np.zeros(3, dtype=np.float64)
            if (
                bool(self.motion_policy.enable_angular)
                and float(np.linalg.norm(rotvec_error))
                > float(self.motion_policy.angular_speed_deadband)
            ):
                angular_cmd = float(self.motion_policy.angular_gain) * rotvec_error
                omega_norm_raw = float(np.linalg.norm(angular_cmd))
                if omega_norm_raw > float(self.motion_policy.angular_speed_max):
                    angular_cmd = angular_cmd * (
                        float(self.motion_policy.angular_speed_max)
                        / max(omega_norm_raw, 1e-12)
                    )
            self.publish_motion_components(
                linear_cmd=linear_cmd,
                angular_cmd=angular_cmd,
                stamp=pose_state.stamp,
            )
            with self.stats_lock:
                self.stats.servo_nonzero += 1
            self._planner_log_throttle(
                "servo_cmd_diag",
                1.0,
                "debug",
                (
                    "servo_cmd: planner_rate=%.1f servo_rate=%.1f spherical_cmd_age_ms=%.1f "
                    "theta_phi_cmd=(%.6f, %.6f) cartesian_cmd_norm=%.6f sat_ratio=%.3f fallback_reason=%s"
                ),
                float(self.planner_hz),
                float(self.servo_hz),
                cmd_age * 1000.0,
                theta_rate,
                phi_rate,
                float(np.linalg.norm(linear_cmd)),
                1.0 - float(linear_scale),
                "none",
            )
        except Exception as exc:
            self._planner_log_throttle(
                "servo_exception",
                1.0,
                "warning",
                "servo 执行异常，发布零速度：%s",
                exc,
            )
            self.publish_zero_twist(pose_state.stamp)
            with self.stats_lock:
                self.stats.servo_zero += 1
                self.stats.servo_zero_missing_cmd += 1

    def _apply_fixed_hemisphere_reference(self) -> None:
        """Temporary override: lock hemisphere center/radius to configured constants."""
        if self.fixed_hemisphere_center is None:
            return
        center_np = np.asarray(self.fixed_hemisphere_center, dtype=np.float64).reshape(3)
        radius = float(self.fixed_hemisphere_radius_m)

        with self.motion_policy_lock:
            self.motion_policy.reference_scene_center = center_np.copy()
            self.motion_policy.reference_radius = radius
            self.motion_policy.reference_initialized = True

        gs_live = getattr(self.omni, "gs", None)
        if gs_live is not None:
            try:
                gs_live.sence_center = torch.as_tensor(center_np, dtype=torch.float32)
            except Exception:
                pass

    def _get_live_scene_center_np(self) -> Optional[np.ndarray]:
        """Try to read a more trustworthy scene center from live backend."""
        if self.fixed_hemisphere_center is not None:
            return np.asarray(self.fixed_hemisphere_center, dtype=np.float64).reshape(3)
        gs_live = getattr(self.omni, "gs", None)
        if gs_live is None:
            return None

        tsdfs = getattr(gs_live, "tsdfs", None)
        if tsdfs is not None and hasattr(tsdfs, "get_pointcloud_center"):
            try:
                center = tsdfs.get_pointcloud_center()
                if center is not None:
                    if isinstance(center, torch.Tensor):
                        center_np = center.detach().cpu().numpy().astype(np.float64).reshape(3)
                    else:
                        center_np = np.asarray(center, dtype=np.float64).reshape(3)
                    if np.isfinite(center_np).all():
                        return center_np
            except Exception:
                pass

        if hasattr(gs_live, "get_fisher_scene_center"):
            try:
                center = gs_live.get_fisher_scene_center()
                if center is not None:
                    if isinstance(center, torch.Tensor):
                        center_np = center.detach().cpu().numpy().astype(np.float64).reshape(3)
                    else:
                        center_np = np.asarray(center, dtype=np.float64).reshape(3)
                    if np.isfinite(center_np).all():
                        return center_np
            except Exception:
                pass
        return None

    def _maybe_recover_reference_radius(
        self,
        *,
        snapshot: PlannerSnapshot,
        pose_state: PoseState,
    ) -> None:
        """Recover from a degenerate tiny reference radius initialized too early."""
        backend = snapshot.backend
        if not hasattr(backend, "get_fisher_scene_center"):
            return
        center = backend.get_fisher_scene_center()
        if center is None:
            return
        if isinstance(center, torch.Tensor):
            center_np = center.detach().cpu().numpy().astype(np.float64).reshape(3)
        else:
            center_np = np.asarray(center, dtype=np.float64).reshape(3)

        current_pos = np.asarray(pose_state.pose_4x4, dtype=np.float64)[:3, 3]
        current_radius = float(np.linalg.norm(current_pos - center_np))
        if not np.isfinite(current_radius):
            return

        with self.motion_policy_lock:
            ref_initialized = bool(getattr(self.motion_policy, "reference_initialized", False))
            ref_radius = getattr(self.motion_policy, "reference_radius", None)
            if not ref_initialized or ref_radius is None:
                return
            ref_radius = float(ref_radius)
            if ref_radius >= self.min_valid_reference_radius_m:
                return

        if current_radius >= self.min_valid_reference_radius_m:
            with self.motion_policy_lock:
                self.motion_policy.reference_initialized = False
                self.motion_policy.reference_radius = None
                self.motion_policy.reference_scene_center = None
            self.planner_logger.warning(
                ("检测到参考半径异常过小（ref=%.4fm），按当前快照重置参考几何；current_radius=%.4fm model_v=%d"),
                ref_radius,
                current_radius,
                int(self.model_version),
            )
            return

        live_center_np = self._get_live_scene_center_np()
        if live_center_np is not None:
            live_radius = float(np.linalg.norm(current_pos - live_center_np))
            if np.isfinite(live_radius) and live_radius >= self.min_valid_reference_radius_m:
                with self.motion_policy_lock:
                    self.motion_policy.reference_scene_center = live_center_np.copy()
                    self.motion_policy.reference_radius = float(live_radius)
                    self.motion_policy.reference_initialized = True
                self.planner_logger.warning(
                    ("参考半径异常（ref=%.4fm, snapshot_r=%.4fm），已使用 live 场景中心强制重置为 %.4fm；model_v=%d"),
                    ref_radius,
                    current_radius,
                    live_radius,
                    int(self.model_version),
                )
                return

        def _vec3_fmt(vec: Optional[np.ndarray]) -> str:
            if vec is None:
                return "None"
            arr = np.asarray(vec, dtype=np.float64).reshape(3)
            if not np.isfinite(arr).all():
                return f"[{arr[0]:.4f}, {arr[1]:.4f}, {arr[2]:.4f}] (non-finite)"
            return f"[{arr[0]:.4f}, {arr[1]:.4f}, {arr[2]:.4f}]"

        self._planner_log_throttle(
            "reference_radius_stuck",
            2.0,
            "warning",
            ("参考半径持续过小且无法恢复：ref=%.4fm snapshot_r=%.4fm min_valid=%.3fm snapshot_center=%s live_center=%s current_pos=%s (请检查 TF 坐标系与 scene_center 来源, world=%s camera=%s)"),
            ref_radius,
            current_radius,
            float(self.min_valid_reference_radius_m),
            _vec3_fmt(center_np),
            _vec3_fmt(live_center_np),
            _vec3_fmt(current_pos),
            self.world_frame,
            self.camera_frame,
        )

    def get_pose_from_tf(self, stamp):
        """
        从 TF 树获取相机 c2w 和 w2c pose 向量（legacy_tf 模式）。
        """
        try:
            transform: TransformStamped = self.tf_buffer.lookup_transform(
                self.world_frame,
                self.camera_frame,
                stamp,
                rospy.Duration(0.1),
            )

            t = transform.transform.translation
            q = transform.transform.rotation

            c2w = np.eye(4, dtype=np.float64)
            c2w[:3, 3] = [t.x, t.y, t.z]
            c2w[:3, :3] = R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()

            w2c = np.linalg.inv(c2w)
            quat = R.from_matrix(w2c[:3, :3]).as_quat()
            pose = np.hstack((w2c[:3, 3], quat))

            return pose, c2w
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as exc:
            stamp_sec = float(stamp.to_sec()) if stamp is not None else 0.0
            now_sec = float(rospy.Time.now().to_sec())
            self._log_throttle(
                "tf_lookup_failure_detail",
                1.0,
                "warning",
                "位姿查询TF失败：world=%s camera=%s rgb_stamp=%.3f now=%.3f lag=%.3fs err=%s",
                self.world_frame,
                self.camera_frame,
                float(stamp_sec),
                float(now_sec),
                float(now_sec - stamp_sec),
                exc,
            )
            return None, None

    def publish_zero_twist(self, stamp=None):
        msg = TwistStamped()
        msg.header.stamp = stamp if stamp is not None else rospy.Time.now()
        msg.header.frame_id = self.cmd_frame
        self.cmd_pub.publish(msg)

    def publish_motion_result(self, motion_result, stamp=None):
        msg = TwistStamped()
        msg.header.stamp = stamp if stamp is not None else rospy.Time.now()
        msg.header.frame_id = self.cmd_frame

        linear = np.asarray(motion_result.velocity_world, dtype=np.float64).reshape(3)
        angular = np.asarray(motion_result.angular_velocity_world, dtype=np.float64).reshape(3)

        msg.twist.linear.x = float(linear[0])
        msg.twist.linear.y = float(linear[1])
        msg.twist.linear.z = float(linear[2])
        msg.twist.angular.x = float(angular[0])
        msg.twist.angular.y = float(angular[1])
        msg.twist.angular.z = float(angular[2])
        self.cmd_pub.publish(msg)

    def publish_motion_components(self, *, linear_cmd: np.ndarray, angular_cmd: np.ndarray, stamp=None):
        msg = TwistStamped()
        msg.header.stamp = stamp if stamp is not None else rospy.Time.now()
        msg.header.frame_id = self.cmd_frame
        linear = np.asarray(linear_cmd, dtype=np.float64).reshape(3)
        angular = np.asarray(angular_cmd, dtype=np.float64).reshape(3)
        msg.twist.linear.x = float(linear[0])
        msg.twist.linear.y = float(linear[1])
        msg.twist.linear.z = float(linear[2])
        msg.twist.angular.x = float(angular[0])
        msg.twist.angular.y = float(angular[1])
        msg.twist.angular.z = float(angular[2])
        self.cmd_pub.publish(msg)

    def maybe_publish_velocity_live(self, pose_4x4, image_shape, stamp):
        """Legacy path: plan directly from live backend after each track."""
        self._apply_fixed_hemisphere_reference()
        try:
            motion_result = self.motion_policy.next_pose_from_c2w(
                gs_backend=self.omni.gs,
                current_c2w=np.asarray(pose_4x4, dtype=np.float64),
                intrinsics_vec=self.calib,
                image_size=image_shape,
                idx=self.next_track_index,
            )
        except Exception as exc:
            self._planner_log_throttle(
                "legacy_policy_not_ready",
                2.0,
                "warning",
                "Fisher 策略尚未就绪，发布零速度：%s",
                exc,
            )
            self.publish_zero_twist(stamp)
            return

        if motion_result.should_stop:
            self.publish_zero_twist(stamp)
            return

        if self.planner_output_mode == "spherical_rate":
            self._cache_spherical_command(
                motion_result=motion_result,
                stamp=stamp,
                model_version=int(self.model_version),
            )
            self._publish_spherical_diag(motion_result=motion_result, stamp=stamp)
            self._servo_timer_callback(None)
            return

        self.publish_motion_result(motion_result, stamp)

    def maybe_export_fisher_snapshot(self, *, pose_4x4, idx: int, tag: str) -> bool:
        if not self.can_export_fisher_snapshots:
            return False

        latest_viewpoint = getattr(self.omni.gs, "viewpoint", None)
        if latest_viewpoint is None:
            keyviews = getattr(self.omni.gs, "keyviewpoints", None)
            if keyviews:
                latest_viewpoint = keyviews[-1]
        if latest_viewpoint is None:
            self._planner_log_throttle(
                "snapshot_viewpoint_missing",
                2.0,
                "warning",
                "跳过 Fisher 快照导出：最新视角不可用（gs.viewpoint 与 gs.keyviewpoints 均为空）。",
            )
            return False

        export_fn = getattr(self.omni.gs, "export_fisher_snapshot", None)
        if export_fn is None:
            self._planner_log_throttle(
                "snapshot_export_fn_missing",
                2.0,
                "warning",
                "跳过 Fisher 快照导出：gs backend 未提供 export_fisher_snapshot()。",
            )
            return False

        try:
            export_fn(
                viewpoint=latest_viewpoint,
                pose=np.asarray(pose_4x4, dtype=np.float64),
                idx=int(idx),
                tag=tag,
            )
            return True
        except Exception as exc:
            self.planner_logger.warning("导出 Fisher 快照失败（%s）：%s", tag, exc)
            return False

    def legacy_image_callback(self, rgb_msg, depth_msg):
        if self.shutdown_requested:
            return
        if self._tracked_frame_limit_reached():
            self.request_shutdown(f"达到跟踪帧上限（{self.max_frames}），停止 ROS spin。")
            return

        pose, pose_4x4 = self.get_pose_from_tf(rgb_msg.header.stamp)
        if pose is None:
            self.publish_zero_twist(rgb_msg.header.stamp)
            return

        self._set_latest_pose(rgb_msg.header.stamp, pose_4x4, pose)

        try:
            rgb_image, depth_image = self._decode_compressed_rgbd(rgb_msg, depth_msg)
        except Exception as exc:
            self.main_logger.error("压缩图像解码失败：%s", exc)
            self.publish_zero_twist(rgb_msg.header.stamp)
            return

        idx = self.next_track_index
        task = self._build_track_task(
            idx=idx,
            stamp=rgb_msg.header.stamp,
            pose_w2c=pose,
            pose_4x4=pose_4x4,
            rgb_image=rgb_image,
            depth_image=depth_image,
            source="legacy_tf",
        )

        if self.should_record_frames:
            self.all_inputs.append(
                {
                    "index": task.index,
                    "image": task.image[None].cpu().numpy(),
                    "depth": task.depth[None].cpu().numpy(),
                    "pose": torch.as_tensor(task.pose_w2c)[None].cpu().numpy(),
                    "intrinsics": self.intrinsics[None].cpu().numpy(),
                    "pose_44": torch.as_tensor(task.pose_4x4)[None].cpu().numpy(),
                    "is_last": False,
                    "depth_scale": self.depth_scale,
                }
            )

        try:
            self.omni.track(
                task.index,
                task.image[None],
                task.depth[None],
                torch.as_tensor(task.pose_w2c)[None],
                self.progress_bar,
                intrinsics=self.intrinsics[None],
                is_last=False,
                pose_44=torch.as_tensor(task.pose_4x4)[None],
                update_rate=self.log_every,
            )
        except Exception as exc:
            self.main_logger.error("跟踪阶段 OMNI.track 执行失败：%s", exc)
            self.publish_zero_twist(task.stamp)
            with self.stats_lock:
                self.stats.track_fail += 1
            return

        with self.stats_lock:
            self.stats.track_success += 1
            tracked_ok = int(self.stats.track_success)

        self.last_pose_4x4 = np.asarray(task.pose_4x4, dtype=np.float64)
        if not self.first_snapshot_saved:
            self.first_snapshot_saved = self.maybe_export_fisher_snapshot(
                pose_4x4=task.pose_4x4,
                idx=task.index,
                tag="first",
            )

        self.maybe_publish_velocity_live(
            pose_4x4=task.pose_4x4,
            image_shape=task.image_shape,
            stamp=task.stamp,
        )

        if should_log_step(task.index, self.log_every):
            self.main_logger.info(
                "legacy 帧=%d pose=(%.3f, %.3f, %.3f) depth_valid_ratio=%.3f",
                int(task.index),
                float(task.pose_4x4[0, 3]),
                float(task.pose_4x4[1, 3]),
                float(task.pose_4x4[2, 3]),
                float(task.depth_valid_ratio),
            )

        self.next_track_index += 1
        if tracked_ok >= self.max_frames:
            self.request_shutdown(f"达到跟踪帧上限（{self.max_frames}），停止 ROS spin。")

    def request_shutdown(self, reason):
        if self.shutdown_requested:
            return
        self.shutdown_requested = True
        self.main_logger.info("%s", reason)
        self.publish_zero_twist()

        if self.mode == "tf_native":
            if self.planner_timer is not None:
                self.planner_timer.shutdown()
            if self.servo_timer is not None:
                self.servo_timer.shutdown()
            if self.track_stop_event is not None:
                self.track_stop_event.set()
            if self.track_queue is not None:
                self.track_queue.close()

        rospy.signal_shutdown(reason)

    def terminate(self):
        self.publish_zero_twist()
        if self.planner_timer is not None:
            self.planner_timer.shutdown()
        if self.servo_timer is not None:
            self.servo_timer.shutdown()

        if self.mode == "tf_native":
            if self.track_stop_event is not None:
                self.track_stop_event.set()
            if self.track_queue is not None:
                self.track_queue.close()
            if self.track_worker_thread is not None:
                self.track_worker_thread.join(timeout=3.0)
                if self.track_worker_thread.is_alive():
                    self.main_logger.warning("跟踪线程未在超时时间内正常退出。")

        self.progress_bar.close()
        last_pose = self.last_pose_4x4
        if self.next_track_index > 0 and last_pose is not None:
            self.maybe_export_fisher_snapshot(
                pose_4x4=last_pose,
                idx=max(self.next_track_index - 1, 0),
                tag="last",
            )

        if self.terminate_enabled:
            if self.should_record_frames and self.all_inputs:
                save_trajectory(self.omni, self.all_inputs, self.output)
            self.omni.terminate()

        with self.stats_lock:
            stats = self.stats
        self.main_logger.info(
            (
                "流水线统计：pose_updates=%d candidates=%d enqueued=%d dropped=%d "
                "track_ok=%d track_fail=%d snapshot_ok=%d snapshot_fail=%d "
                "planner_steps=%d planner_nonzero=%d planner_zero=%d "
                "planner_zero_missing=%d planner_zero_stale=%d planner_zero_stop=%d planner_zero_exc=%d "
                "servo_steps=%d servo_nonzero=%d servo_zero=%d "
                "servo_zero_missing_cmd=%d servo_zero_cmd_stale=%d servo_zero_pose_stale=%d "
                "gate_passed=%d gate_interval=%d gate_motion=%d gate_forced=%d"
            ),
            stats.pose_updates,
            stats.frame_candidates,
            stats.frames_enqueued,
            stats.queue_dropped,
            stats.track_success,
            stats.track_fail,
            stats.snapshot_success,
            stats.snapshot_fail,
            stats.planner_steps,
            stats.planner_nonzero,
            stats.planner_zero,
            stats.planner_zero_missing_input,
            stats.planner_zero_pose_stale,
            stats.planner_zero_policy_stop,
            stats.planner_zero_exception,
            stats.servo_steps,
            stats.servo_nonzero,
            stats.servo_zero,
            stats.servo_zero_missing_cmd,
            stats.servo_zero_cmd_stale,
            stats.servo_zero_pose_stale,
            stats.gated_passed,
            stats.gated_by_interval,
            stats.gated_by_motion,
            stats.forced_gap_keyframes,
        )

        rospy.signal_shutdown("InfoFlow ROS 节点已终止")


def build_argparser():
    parser = argparse.ArgumentParser(
        description=("InfoFlow ROS Node\n默认采用 TF 驱动 + Tracking/Planning 解耦。\n也支持 legacy_tf 回滚模式（串行 TF->track->plan）。\n"),
        epilog=(
            "tf_native 示例：\n"
            "  python info_flow/info_flow_node.py \\\n"
            "      --config config/rtabmap_config.yaml \\\n"
            "      --slam_frontend_mode tf_native \\\n"
            "      --world_frame base_link --camera_frame cam_1_color_optical_frame \\\n"
            "      --planner_hz 30 --pose_stale_timeout_sec 0.2 \\\n"
            "      --track_queue_size 2 --max_frames 500\n\n"
            "legacy_tf 回滚示例：\n"
            "  python info_flow/info_flow_node.py \\\n"
            "      --config config/rtabmap_config.yaml \\\n"
            "      --slam_frontend_mode legacy_tf\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="config/rtabmap_config.yaml",
        help="config file path",
    )
    parser.add_argument(
        "--slam_frontend_mode",
        type=str,
        default="tf_native",
        choices=("tf_native", "rtabmap_native", "legacy_tf"),
        help="frontend mode: tf_native (default), rtabmap_native(alias to tf_native), or legacy_tf (rollback)",
    )
    parser.add_argument(
        "-r",
        "--rgb_topic",
        type=str,
        default="/cam_1/color/image_raw/compressed",
        help="RGB compressed image topic",
    )
    parser.add_argument(
        "-d",
        "--depth_topic",
        type=str,
        default="/cam_1/aligned_depth_to_color/image_raw/compressed",
        help="Depth compressed image topic",
    )
    parser.add_argument(
        "--camera_info_topic",
        type=str,
        default="/cam_1/color/camera_info",
        help="camera info topic",
    )
    parser.add_argument(
        "--slam_odom_topic",
        type=str,
        default="/rtabmap/odom",
        help="[deprecated] kept for compatibility; ignored in tf_native mode",
    )
    parser.add_argument(
        "--odom_info_topic",
        type=str,
        default="/rtabmap/odom_info",
        help="[deprecated] kept for compatibility; ignored in tf_native mode",
    )
    parser.add_argument(
        "--world_frame",
        type=str,
        default="base_link",
        help="world frame id used in tf_native / legacy_tf mode",
    )
    parser.add_argument(
        "--camera_frame",
        type=str,
        default="cam_1_color_optical_frame",
        help="camera frame id used in tf_native / legacy_tf mode",
    )
    parser.add_argument(
        "--cmd_topic",
        type=str,
        default="/servo_server/delta_twist_camera",
        help="output TwistStamped topic",
    )
    parser.add_argument(
        "--cmd_frame",
        type=str,
        default="base_link",
        help="frame_id used for TwistStamped commands",
    )
    parser.add_argument(
        "--planner_hz",
        type=float,
        default=30.0,
        help="planning loop frequency (Hz) for tf_native mode",
    )
    parser.add_argument(
        "--servo_hz",
        type=float,
        default=50.0,
        help="servo publish loop frequency (Hz) used in spherical_rate mode",
    )
    parser.add_argument(
        "--planner_output_mode",
        type=str,
        default="cartesian_legacy",
        choices=("cartesian_legacy", "spherical_rate"),
        help="planner output mode: legacy cartesian Twist or spherical rate command",
    )
    parser.add_argument(
        "--spherical_cmd_timeout_sec",
        type=float,
        default=0.25,
        help="timeout for cached spherical command before forcing zero Twist",
    )
    parser.add_argument(
        "--pose_stale_timeout_sec",
        type=float,
        default=0.2,
        help="if latest corrected pose is older than this threshold, publish zero twist",
    )
    parser.add_argument(
        "--track_queue_size",
        type=int,
        default=2,
        help="bounded tracking queue size, oldest tasks are dropped on overflow",
    )
    parser.add_argument(
        "--keyframe_min_interval_sec",
        type=float,
        default=0.10,
        help="minimum timestamp interval between accepted keyframes",
    )
    parser.add_argument(
        "--keyframe_min_translation_m",
        type=float,
        default=0.01,
        help="minimum translation increment between accepted keyframes",
    )
    parser.add_argument(
        "--keyframe_min_rotation_deg",
        type=float,
        default=1.0,
        help="minimum rotation increment (deg) between accepted keyframes",
    )
    parser.add_argument(
        "--sync_slop_sec",
        type=float,
        default=0.12,
        help="approximate time synchronizer slop (sec) for RGBD streams",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=f"replica/output/{time.strftime('%Y%m%d_%H%M%S')}",
        help="output path",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=500,
        help="maximum number of tracked frames to process before shutdown",
    )
    parser.add_argument(
        "--terminate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="run post-processing on exit, including OMNI termination and optional frame saving",
    )
    parser.add_argument(
        "--save_fisher_snapshots",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="save first/last Fisher point-cloud snapshots under output/nbv_vis",
    )
    parser.add_argument(
        "--depth_scale",
        type=float,
        default=1000.0,
        help="depth scale factor",
    )
    parser.add_argument(
        "--fisher_step_scale",
        type=float,
        default=1e-5,
        help="shared Fisher control scale for theta/phi",
    )
    parser.add_argument(
        "--linear_vel_max",
        type=float,
        default=0.05,
        help="maximum Cartesian linear velocity",
    )
    parser.add_argument(
        "--angular_gain",
        type=float,
        default=2.0,
        help="angular gain for omega command",
    )
    parser.add_argument(
        "--radial_gain",
        type=float,
        default=0.2,
        help="radial correction gain",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=1,
        help="control timestep used by FisherMotionPolicy",
    )
    parser.add_argument(
        "--grad_eps",
        type=float,
        default=0.01,
        help="finite difference epsilon for Fisher gradients",
    )
    parser.add_argument(
        "--spherical_speed_min",
        type=float,
        default=0.0,
        help="minimum spherical speed before publishing zero twist",
    )
    parser.add_argument(
        "--enable_angular",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enable angular velocity output",
    )
    parser.add_argument(
        "--angular_speed_max",
        type=float,
        default=1.0,
        help="maximum norm of cartesian angular velocity command (rad/s)",
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
        help="Emit frame-level summary every N frames.",
    )
    parser.add_argument(
        "--status_log_interval_sec",
        type=float,
        default=1.0,
        help="Emit planning/queue status summary every N seconds.",
    )
    parser.add_argument(
        "--log_file",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable or disable run.log output in output directory.",
    )
    parser.add_argument(
        "--vis_gui",
        action="store_true",
        help="use opencv to visuliazation the whole process",
    )
    return parser


if __name__ == "__main__":
    parser = build_argparser()
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    log_file_path = os.path.join(args.output, "run.log") if bool(args.log_file) else None
    requested_sections = args.log_section or ["all"]
    selected_sections = None if "all" in {str(s).lower() for s in requested_sections} else requested_sections
    configure_logging(
        profile=str(args.log_profile),
        level=args.log_level,
        log_file=log_file_path,
        enabled_sections=selected_sections,
        min_console_level=args.log_min_level,
        force=True,
    )
    config = load_config(args.config)
    # Default Fisher visualization sampling density for this entrypoint.
    config.setdefault("fisher_num_samples", 128)
    config.setdefault("fisher_num_dense_points", 1024)
    try:
        torch.multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass

    node = InfoFlowROSNode(args, config)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        get_section_logger("entry.infoflow", "main").info("收到中断信号，开始关闭流程...")
    finally:
        node.terminate()
        get_section_logger("entry.infoflow", "main").info("运行结束，结果已保存至 %s", node.output)
