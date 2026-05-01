from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import rospy
import tf2_ros
from scipy.spatial.transform import Rotation as R

from distributed_common import (
    CachedSphericalCommand,
    configure_entry_logging,
    import_omnimap_msgs,
    local_frame_from_theta_phi,
    load_runtime_config,
    lookup_latest_pose_from_tf,
    look_at_c2w,
    position_to_spherical,
    publish_motion_components,
    publish_zero_twist,
    set_nofile_limit,
)
from geometry_msgs.msg import TwistStamped
from omnimap.util.utils import get_section_logger


@dataclass
class ServoStats:
    servo_steps: int = 0
    servo_nonzero: int = 0
    servo_zero_missing_cmd: int = 0
    servo_zero_cmd_stale: int = 0
    servo_zero_pose_stale: int = 0
    servo_zero_tf_fail: int = 0
    servo_zero_policy_stop: int = 0
    servo_exception: int = 0


class InfoFlowServoNode:
    def __init__(self, args, config):
        rospy.init_node("info_flow_servo_node", anonymous=True)
        _, SphericalCommandMsg = import_omnimap_msgs()

        self.args = args
        self.config = config
        self.SphericalCommandMsg = SphericalCommandMsg
        self.main_logger = get_section_logger("entry.infoflow_servo", "main")
        self.planner_logger = get_section_logger("planner.infoflow_servo", "planner")
        self.profile_logger = get_section_logger("profile.infoflow_servo", "profile")
        self.world_frame = str(args.world_frame)
        self.camera_frame = str(args.camera_frame)
        self.cmd_frame = str(args.cmd_frame)
        self.servo_hz = float(args.servo_hz)
        self.pose_stale_timeout_sec = float(args.pose_stale_timeout_sec)
        self.spherical_cmd_timeout_sec = float(args.spherical_cmd_timeout_sec)
        self.linear_vel_max = float(args.linear_vel_max)
        self.angular_speed_max = float(args.angular_speed_max)
        self.enable_angular = bool(args.enable_angular)
        self.angular_speed_deadband = 1e-3
        self.status_log_interval_sec = max(0.2, float(args.status_log_interval_sec))

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.cmd_pub = rospy.Publisher(args.cmd_topic, TwistStamped, queue_size=1)
        self.latest_cmd: Optional[CachedSphericalCommand] = None

        self.cmd_sub = rospy.Subscriber(
            args.spherical_cmd_topic,
            SphericalCommandMsg,
            self.spherical_cmd_callback,
            queue_size=1,
        )
        self.timer = rospy.Timer(
            rospy.Duration(1.0 / max(self.servo_hz, 1e-6)),
            self._servo_timer_callback,
        )

        self.stats = ServoStats()
        self._last_status_wall = float(time.monotonic())
        self._last_status_steps = 0
        self._last_status_nonzero = 0
        self._throttle_last = {}

        self.main_logger.info(
            "Servo 节点已启动：spherical_cmd_topic=%s cmd_topic=%s servo_hz=%.1f",
            args.spherical_cmd_topic,
            args.cmd_topic,
            float(self.servo_hz),
        )

    def _planner_log_throttle(self, key: str, interval_sec: float, level: str, msg: str, *args):
        now = float(time.monotonic())
        last = float(self._throttle_last.get(key, -1e18))
        if (now - last) < float(interval_sec):
            return
        self._throttle_last[key] = now
        getattr(self.planner_logger, str(level).lower())(msg, *args)

    def spherical_cmd_callback(self, msg) -> None:
        self.latest_cmd = CachedSphericalCommand(
            msg=msg,
            receipt_wall_time=float(time.monotonic()),
        )
        self.profile_logger.debug(
            "cmd_rx: model_v=%d tick=%d stop=%s reason=%s",
            int(msg.model_version),
            int(msg.planner_tick),
            bool(msg.should_stop),
            str(msg.stop_reason),
        )

    def _maybe_log_status(self) -> None:
        now = float(time.monotonic())
        elapsed = now - self._last_status_wall
        if elapsed < self.status_log_interval_sec:
            return
        stats = ServoStats(**self.stats.__dict__)
        d_steps = int(stats.servo_steps) - int(self._last_status_steps)
        d_nonzero = int(stats.servo_nonzero) - int(self._last_status_nonzero)
        self._last_status_wall = now
        self._last_status_steps = int(stats.servo_steps)
        self._last_status_nonzero = int(stats.servo_nonzero)
        self.planner_logger.info(
            (
                "Servo 状态：servo_hz=%.1f cmd_hz=%.1f 累计(steps=%d nonzero=%d "
                "missing_cmd=%d cmd_stale=%d pose_stale=%d tf_fail=%d policy_stop=%d exc=%d)"
            ),
            float(d_steps / max(elapsed, 1e-6)),
            float(d_nonzero / max(elapsed, 1e-6)),
            int(stats.servo_steps),
            int(stats.servo_nonzero),
            int(stats.servo_zero_missing_cmd),
            int(stats.servo_zero_cmd_stale),
            int(stats.servo_zero_pose_stale),
            int(stats.servo_zero_tf_fail),
            int(stats.servo_zero_policy_stop),
            int(stats.servo_exception),
        )

    def _servo_timer_callback(self, _event) -> None:
        self.stats.servo_steps += 1

        cmd_cache = self.latest_cmd
        if cmd_cache is None:
            publish_zero_twist(self.cmd_pub, cmd_frame=self.cmd_frame)
            self.stats.servo_zero_missing_cmd += 1
            self._maybe_log_status()
            return

        cmd = cmd_cache.msg
        cmd_age = float(time.monotonic() - cmd_cache.receipt_wall_time)
        if cmd_age > self.spherical_cmd_timeout_sec:
            self._planner_log_throttle(
                "cmd_stale",
                1.0,
                "warning",
                "servo 命令过期（age=%.3fs > %.3fs），发布零速度。",
                cmd_age,
                self.spherical_cmd_timeout_sec,
            )
            publish_zero_twist(self.cmd_pub, cmd_frame=self.cmd_frame)
            self.stats.servo_zero_cmd_stale += 1
            self._maybe_log_status()
            return

        try:
            tf_t0 = time.perf_counter()
            _, current_c2w, pose_stamp = lookup_latest_pose_from_tf(
                self.tf_buffer,
                world_frame=self.world_frame,
                camera_frame=self.camera_frame,
            )
            self.profile_logger.debug(
                "tf_lookup: pose_ms=%.2f stamp=%.3f",
                float((time.perf_counter() - tf_t0) * 1000.0),
                float(pose_stamp.to_sec()),
            )
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as exc:
            self._planner_log_throttle(
                "tf_failure",
                1.0,
                "warning",
                "servo 本机 TF 查询失败，发布零速度：%s",
                exc,
            )
            publish_zero_twist(self.cmd_pub, cmd_frame=self.cmd_frame)
            self.stats.servo_zero_tf_fail += 1
            self._maybe_log_status()
            return

        pose_age = float((rospy.Time.now() - pose_stamp).to_sec())
        if pose_age > self.pose_stale_timeout_sec:
            self._planner_log_throttle(
                "pose_stale",
                1.0,
                "warning",
                "servo 位姿过期（age=%.3fs > %.3fs），发布零速度。",
                pose_age,
                self.pose_stale_timeout_sec,
            )
            self.profile_logger.info(
                "pose_stale(servo): pose_age=%.3fs cmd_age=%.3fs pose_stamp=%.3f",
                float(pose_age),
                float(cmd_age),
                float(pose_stamp.to_sec()),
            )
            publish_zero_twist(self.cmd_pub, cmd_frame=self.cmd_frame, stamp=pose_stamp)
            self.stats.servo_zero_pose_stale += 1
            self._maybe_log_status()
            return

        if bool(cmd.should_stop):
            publish_zero_twist(self.cmd_pub, cmd_frame=self.cmd_frame, stamp=pose_stamp)
            self.stats.servo_zero_policy_stop += 1
            self._maybe_log_status()
            return

        try:
            current_position = np.asarray(current_c2w[:3, 3], dtype=np.float64)
            reference_scene_center = np.asarray(cmd.reference_scene_center, dtype=np.float64).reshape(3)
            radius, theta, phi = position_to_spherical(current_position, reference_scene_center)
            e_theta, e_phi, _ = local_frame_from_theta_phi(theta, phi)
            theta_rate = float(cmd.delta_theta)
            phi_rate = float(cmd.delta_phi)
            linear_cmd = radius * (theta_rate * e_theta + phi_rate * e_phi)
            linear_norm_raw = float(np.linalg.norm(linear_cmd))
            if linear_norm_raw > self.linear_vel_max:
                linear_cmd = linear_cmd * (self.linear_vel_max / max(linear_norm_raw, 1e-12))

            desired_c2w = look_at_c2w(current_position, reference_scene_center)
            current_rotation = np.asarray(current_c2w[:3, :3], dtype=np.float64)
            desired_rotation = np.asarray(desired_c2w[:3, :3], dtype=np.float64)
            rotation_error = desired_rotation @ current_rotation.T
            rotvec_error = R.from_matrix(rotation_error).as_rotvec().astype(np.float64)
            angular_cmd = np.zeros(3, dtype=np.float64)
            if self.enable_angular and float(np.linalg.norm(rotvec_error)) > self.angular_speed_deadband:
                angular_cmd = rotvec_error / max(float(cmd.dt), 1e-6)
                omega_norm_raw = float(np.linalg.norm(angular_cmd))
                if omega_norm_raw > self.angular_speed_max:
                    angular_cmd = angular_cmd * (
                        self.angular_speed_max / max(omega_norm_raw, 1e-12)
                    )

            publish_motion_components(
                self.cmd_pub,
                cmd_frame=self.cmd_frame,
                linear_cmd=linear_cmd,
                angular_cmd=angular_cmd,
                stamp=pose_stamp,
            )
            self.profile_logger.debug(
                "servo_cmd: cmd_age_ms=%.1f theta_phi=(%.6f, %.6f) linear_norm=%.6f angular_norm=%.6f",
                float(cmd_age * 1000.0),
                float(theta_rate),
                float(phi_rate),
                float(np.linalg.norm(linear_cmd)),
                float(np.linalg.norm(angular_cmd)),
            )
            self.stats.servo_nonzero += 1
        except Exception as exc:
            self._planner_log_throttle(
                "servo_exception",
                1.0,
                "warning",
                "servo 执行异常，发布零速度：%s",
                exc,
            )
            publish_zero_twist(self.cmd_pub, cmd_frame=self.cmd_frame, stamp=pose_stamp)
            self.stats.servo_exception += 1
        self._maybe_log_status()


def build_argparser():
    parser = argparse.ArgumentParser(
        description="Distributed InfoFlow servo node (execution side).",
    )
    parser.add_argument("-c", "--config", type=str, default="config/rtabmap_config.yaml")
    parser.add_argument("--spherical_cmd_topic", type=str, default="/omnimap/spherical_cmd")
    parser.add_argument("--cmd_topic", type=str, default="/servo_server/delta_twist_camera")
    parser.add_argument("--cmd_frame", type=str, default="base_link")
    parser.add_argument("--world_frame", type=str, default="base_link")
    parser.add_argument("--camera_frame", type=str, default="cam_1_color_optical_frame")
    parser.add_argument("--servo_hz", type=float, default=50.0)
    parser.add_argument("--spherical_cmd_timeout_sec", type=float, default=0.25)
    parser.add_argument("--pose_stale_timeout_sec", type=float, default=0.2)
    parser.add_argument("--linear_vel_max", type=float, default=0.05)
    parser.add_argument("--angular_speed_max", type=float, default=1.0)
    parser.add_argument("--enable_angular", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("-o", "--output", type=str, default=f"replica/output/{time.strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--log_profile", choices=("quiet", "default", "debug"), default="default")
    parser.add_argument("--log_level", type=str, default=None)
    parser.add_argument(
        "--log_section",
        action="append",
        choices=("all", "main", "tsdf", "gaussian", "fisher", "planner", "profile"),
        default=None,
    )
    parser.add_argument("--log_min_level", choices=("DEBUG", "INFO", "WARNING"), default="INFO")
    parser.add_argument("--status_log_interval_sec", type=float, default=1.0)
    parser.add_argument(
        "--log_file",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


if __name__ == "__main__":
    parser = build_argparser()
    args = parser.parse_args()
    set_nofile_limit()
    configure_entry_logging(args)
    config = load_runtime_config(args.config)
    node = InfoFlowServoNode(args, config)
    rospy.spin()
