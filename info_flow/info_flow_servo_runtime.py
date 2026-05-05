from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import rospy
import tf2_ros
from geometry_msgs.msg import TwistStamped
from scipy.spatial.transform import Rotation as R

from servo_runtime_common import (
    configure_servo_logging,
    get_section_logger,
    import_spherical_command_msg,
    load_servo_motion_config,
    local_frame_from_theta_phi,
    lookup_latest_pose_from_tf,
    look_at_c2w,
    position_to_spherical,
    publish_motion_components,
    publish_zero_twist,
    set_nofile_limit,
)


@dataclass
class CachedSphericalCommand:
    msg: object
    receipt_wall_time: float


@dataclass
class ServoStats:
    servo_steps: int = 0
    servo_nonzero: int = 0
    cmd_rx_total: int = 0
    servo_zero_missing_cmd: int = 0
    servo_zero_cmd_stale: int = 0
    servo_zero_pose_stale: int = 0
    servo_zero_tf_fail: int = 0
    servo_zero_policy_stop: int = 0
    servo_exception: int = 0


class InfoFlowServoRuntime:
    def __init__(self, args):
        rospy.init_node("info_flow_servo_runtime", anonymous=True)
        spherical_command_msg = import_spherical_command_msg()

        self.main_logger = get_section_logger("entry.infoflow_servo_runtime", "main")
        self.planner_logger = get_section_logger("planner.infoflow_servo_runtime", "planner")
        self.profile_logger = get_section_logger("profile.infoflow_servo_runtime", "profile")
        self.world_frame = str(args.world_frame)
        self.camera_frame = str(args.camera_frame)
        self.cmd_frame = str(args.cmd_frame)
        self.servo_hz = float(args.servo_hz)
        self.pose_stale_timeout_sec = float(args.pose_stale_timeout_sec)
        self.spherical_cmd_timeout_sec = float(args.spherical_cmd_timeout_sec)
        self.adaptive_cmd_timeout_scale = max(1.0, float(args.adaptive_cmd_timeout_scale))
        self.adaptive_cmd_timeout_cap_sec = max(
            self.spherical_cmd_timeout_sec,
            float(args.adaptive_cmd_timeout_cap_sec),
        )
        self.linear_vel_max = float(args.linear_vel_max)
        self.angular_speed_max = float(args.angular_speed_max)
        self.enable_angular = bool(args.enable_angular)
        self.radial_gain = float(args.radial_gain)
        self.radial_i_gain = max(0.0, float(args.radial_i_gain))
        self.radial_deadband = max(0.0, float(args.radial_deadband))
        self.radial_integral_limit = max(0.0, float(args.radial_integral_limit))
        self.radial_vel_max = max(0.0, float(args.radial_vel_max))
        self.angular_gain = float(args.angular_gain)
        self.angular_deadband = max(0.0, float(args.angular_deadband))
        self.linear_accel_max = max(0.0, float(args.linear_accel_max))
        self.angular_accel_max = max(0.0, float(args.angular_accel_max))
        self.status_log_interval_sec = max(0.2, float(args.status_log_interval_sec))

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.cmd_pub = rospy.Publisher(args.cmd_topic, TwistStamped, queue_size=1)
        self.latest_cmd: Optional[CachedSphericalCommand] = None
        self._last_cmd_receipt_wall: Optional[float] = None
        self._cmd_interval_ema_sec: Optional[float] = None
        self._last_linear_cmd = np.zeros(3, dtype=np.float64)
        self._last_angular_cmd = np.zeros(3, dtype=np.float64)
        self._last_servo_wall: Optional[float] = None
        self._radial_integral_error = 0.0

        self.cmd_sub = rospy.Subscriber(
            args.spherical_cmd_topic,
            spherical_command_msg,
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
            ("轻量 Servo runtime 已启动：spherical_cmd_topic=%s cmd_topic=%s servo_hz=%.1f cmd_timeout=%.3fs adaptive_scale=%.2f adaptive_cap=%.3fs"),
            args.spherical_cmd_topic,
            args.cmd_topic,
            float(self.servo_hz),
            float(self.spherical_cmd_timeout_sec),
            float(self.adaptive_cmd_timeout_scale),
            float(self.adaptive_cmd_timeout_cap_sec),
        )
        self.main_logger.info(
            "ROS 环境：ROS_MASTER_URI=%s ROS_IP=%s ROS_HOSTNAME=%s",
            os.environ.get("ROS_MASTER_URI", ""),
            os.environ.get("ROS_IP", ""),
            os.environ.get("ROS_HOSTNAME", ""),
        )
        self.main_logger.info(
            (
                "Servo 控制参数：linear_vel_max=%.4f radial_gain=%.4f radial_i_gain=%.4f "
                "radial_vel_max=%.4f radial_deadband=%.4f radial_integral_limit=%.4f "
                "angular_gain=%.4f angular_deadband=%.4f angular_speed_max=%.4f "
                "linear_accel_max=%.4f angular_accel_max=%.4f"
            ),
            float(self.linear_vel_max),
            float(self.radial_gain),
            float(self.radial_i_gain),
            float(self.radial_vel_max),
            float(self.radial_deadband),
            float(self.radial_integral_limit),
            float(self.angular_gain),
            float(self.angular_deadband),
            float(self.angular_speed_max),
            float(self.linear_accel_max),
            float(self.angular_accel_max),
        )

    def _planner_log_throttle(self, key: str, interval_sec: float, level: str, msg: str, *args):
        now = float(time.monotonic())
        last = float(self._throttle_last.get(key, -1e18))
        if (now - last) < float(interval_sec):
            return
        self._throttle_last[key] = now
        getattr(self.planner_logger, str(level).lower())(msg, *args)

    @staticmethod
    def _clip_vector_norm(vec: np.ndarray, max_norm: float) -> tuple[np.ndarray, float]:
        vec = np.asarray(vec, dtype=np.float64).reshape(3)
        norm = float(np.linalg.norm(vec))
        limit = float(max_norm)
        if norm <= 1e-12 or limit <= 0.0 or norm <= limit:
            return vec, norm
        clipped = vec * (limit / norm)
        return clipped, float(np.linalg.norm(clipped))

    @staticmethod
    def _slew_limit_vector(
        desired: np.ndarray,
        previous: np.ndarray,
        max_delta_norm: float,
    ) -> np.ndarray:
        desired = np.asarray(desired, dtype=np.float64).reshape(3)
        previous = np.asarray(previous, dtype=np.float64).reshape(3)
        max_delta = float(max_delta_norm)
        if max_delta <= 0.0:
            return desired
        delta = desired - previous
        delta_norm = float(np.linalg.norm(delta))
        if delta_norm <= max_delta or delta_norm <= 1e-12:
            return desired
        return previous + delta * (max_delta / delta_norm)

    @staticmethod
    def _clip_scalar(value: float, limit: float) -> tuple[float, bool]:
        limit = max(0.0, float(limit))
        value = float(value)
        if limit <= 0.0:
            return 0.0, abs(value) > 1e-12
        clipped = float(np.clip(value, -limit, limit))
        return clipped, bool(abs(clipped - value) > 1e-12)

    def _reset_slew_state(self) -> None:
        self._last_linear_cmd = np.zeros(3, dtype=np.float64)
        self._last_angular_cmd = np.zeros(3, dtype=np.float64)
        self._last_servo_wall = float(time.monotonic())
        self._radial_integral_error = 0.0

    def _servo_dt_sec(self) -> float:
        now = float(time.monotonic())
        default_dt = 1.0 / max(float(self.servo_hz), 1e-6)
        if self._last_servo_wall is None:
            dt_sec = default_dt
        else:
            dt_sec = now - float(self._last_servo_wall)
            dt_sec = min(max(dt_sec, 1e-6), 0.1)
        self._last_servo_wall = now
        return float(dt_sec)

    def spherical_cmd_callback(self, msg) -> None:
        receipt_wall_time = float(time.monotonic())
        if self._last_cmd_receipt_wall is not None:
            interval_sec = max(0.0, receipt_wall_time - self._last_cmd_receipt_wall)
            if self._cmd_interval_ema_sec is None:
                self._cmd_interval_ema_sec = interval_sec
            else:
                self._cmd_interval_ema_sec = 0.2 * interval_sec + 0.8 * self._cmd_interval_ema_sec
        self._last_cmd_receipt_wall = receipt_wall_time
        self.latest_cmd = CachedSphericalCommand(
            msg=msg,
            receipt_wall_time=receipt_wall_time,
        )
        self.stats.cmd_rx_total += 1
        self.profile_logger.debug(
            "cmd_rx: model_v=%d tick=%d stop=%s reason=%s ema_interval=%.3fs",
            int(msg.model_version),
            int(msg.planner_tick),
            bool(msg.should_stop),
            str(msg.stop_reason),
            float(self._cmd_interval_ema_sec if self._cmd_interval_ema_sec is not None else -1.0),
        )

    def _effective_cmd_timeout_sec(self) -> float:
        timeout_sec = float(self.spherical_cmd_timeout_sec)
        if self._cmd_interval_ema_sec is None:
            return timeout_sec
        adaptive_timeout = float(self._cmd_interval_ema_sec) * float(self.adaptive_cmd_timeout_scale)
        adaptive_timeout = min(adaptive_timeout, float(self.adaptive_cmd_timeout_cap_sec))
        return max(timeout_sec, adaptive_timeout)

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
                "Servo 状态：servo_hz=%.1f cmd_hz=%.1f "
                "conn(sub=%d pub=%d) 累计(rx=%d steps=%d nonzero=%d missing_cmd=%d cmd_stale=%d pose_stale=%d tf_fail=%d policy_stop=%d exc=%d)"
            ),
            float(d_steps / max(elapsed, 1e-6)),
            float(d_nonzero / max(elapsed, 1e-6)),
            int(self.cmd_sub.get_num_connections()),
            int(self.cmd_pub.get_num_connections()),
            int(stats.cmd_rx_total),
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
        # 周期伺服主循环：
        # 1) 读取最新 planner 命令与本机位姿；
        # 2) 通过多重安全门（缺命令/过期/TF失败/位姿过期/策略要求停止）；
        # 3) 将球坐标速度指令转换为 Twist 并发布。
        self.stats.servo_steps += 1

        cmd_cache = self.latest_cmd
        if cmd_cache is None:
            # 尚未收到任何规划命令：主动发布零速，保证执行侧静止安全。
            publish_zero_twist(self.cmd_pub, cmd_frame=self.cmd_frame)
            self._reset_slew_state()
            self.stats.servo_zero_missing_cmd += 1
            self._maybe_log_status()
            return

        cmd = cmd_cache.msg
        cmd_age = float(time.monotonic() - cmd_cache.receipt_wall_time)
        effective_timeout_sec = self._effective_cmd_timeout_sec()
        if cmd_age > effective_timeout_sec:
            # 命令超时：避免执行陈旧控制量，降级为零速。
            self._planner_log_throttle(
                "cmd_stale",
                1.0,
                "warning",
                "servo 命令过期（age=%.3fs > %.3fs，base=%.3fs ema=%.3fs），发布零速度。",
                cmd_age,
                effective_timeout_sec,
                self.spherical_cmd_timeout_sec,
                float(self._cmd_interval_ema_sec if self._cmd_interval_ema_sec is not None else -1.0),
            )
            publish_zero_twist(self.cmd_pub, cmd_frame=self.cmd_frame)
            self._reset_slew_state()
            self.stats.servo_zero_cmd_stale += 1
            self._maybe_log_status()
            return

        try:
            # 从本机 TF 获取当前相机位姿（执行侧闭环反馈）。
            tf_t0 = time.perf_counter()
            current_c2w, pose_stamp = lookup_latest_pose_from_tf(
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
            self._reset_slew_state()
            self.stats.servo_zero_tf_fail += 1
            self._maybe_log_status()
            return

        pose_age = float((rospy.Time.now() - pose_stamp).to_sec())
        if pose_age > self.pose_stale_timeout_sec:
            # 位姿过期：停止输出运动，防止基于过时状态运动。
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
            self._reset_slew_state()
            self.stats.servo_zero_pose_stale += 1
            self._maybe_log_status()
            return

        if bool(cmd.should_stop):
            # planner 明确要求 stop（如策略收敛/风险条件触发）。
            publish_zero_twist(self.cmd_pub, cmd_frame=self.cmd_frame, stamp=pose_stamp)
            self._reset_slew_state()
            self.stats.servo_zero_policy_stop += 1
            self._maybe_log_status()
            return

        try:
            servo_dt_sec = self._servo_dt_sec()
            current_position = np.asarray(current_c2w[:3, 3], dtype=np.float64)
            reference_scene_center = np.asarray(
                cmd.reference_scene_center,
                dtype=np.float64,
            ).reshape(3)
            radius, theta, phi = position_to_spherical(current_position, reference_scene_center)
            e_theta, e_phi, n_hat = local_frame_from_theta_phi(theta, phi)
            theta_rate = float(cmd.theta_rate)
            phi_rate = float(cmd.phi_rate)
            # 切向速度：沿球面切平面的角速度分量。
            v_t = radius * (theta_rate * e_theta + phi_rate * e_phi)

            reference_radius = float(cmd.reference_radius)
            radial_error = float(reference_radius - radius)
            # 径向 PI：负责把当前半径收敛到参考半径，带抗积分饱和与速度限幅。
            radial_p_term = self.radial_gain * radial_error
            radial_i_term = self.radial_i_gain * self._radial_integral_error
            radial_speed_raw = radial_p_term + radial_i_term
            radial_speed_cmd = 0.0
            radial_limited = False
            radial_integral_accepted = False
            if abs(radial_error) > self.radial_deadband:
                _, radial_limited_pre = self._clip_scalar(
                    radial_speed_raw,
                    self.radial_vel_max,
                )
                anti_windup_allows_integral = not radial_limited_pre or radial_speed_raw * radial_error < 0.0
                if anti_windup_allows_integral:
                    self._radial_integral_error = float(
                        np.clip(
                            self._radial_integral_error + radial_error * servo_dt_sec,
                            -self.radial_integral_limit,
                            self.radial_integral_limit,
                        )
                    )
                    radial_integral_accepted = True
                radial_i_term = self.radial_i_gain * self._radial_integral_error
                radial_speed_raw = radial_p_term + radial_i_term
                radial_speed_cmd, radial_limited = self._clip_scalar(
                    radial_speed_raw,
                    self.radial_vel_max,
                )
                v_r = radial_speed_cmd * n_hat
            else:
                # 在 deadband 内清零积分，避免微小抖动长期累积。
                self._radial_integral_error = 0.0
                radial_i_term = 0.0
                radial_speed_raw = radial_p_term
                v_r = np.zeros(3, dtype=np.float64)

            # 平移速度 = 切向 + 径向，再做速度上限与加速度斜率限制（slew rate）。
            linear_raw = v_t + v_r
            linear_raw_norm = float(np.linalg.norm(linear_raw))
            linear_cmd, linear_limited_norm = self._clip_vector_norm(
                linear_raw,
                self.linear_vel_max,
            )
            linear_cmd = self._slew_limit_vector(
                linear_cmd,
                self._last_linear_cmd,
                self.linear_accel_max * servo_dt_sec,
            )

            desired_c2w = look_at_c2w(current_position, reference_scene_center)
            current_rotation = np.asarray(current_c2w[:3, :3], dtype=np.float64)
            desired_rotation = np.asarray(desired_c2w[:3, :3], dtype=np.float64)
            rotation_error = desired_rotation @ current_rotation.T
            rotvec_error = R.from_matrix(rotation_error).as_rotvec().astype(np.float64)
            angular_cmd = np.zeros(3, dtype=np.float64)
            angular_raw_norm = 0.0
            angular_limited_norm = 0.0
            rotvec_error_norm = float(np.linalg.norm(rotvec_error))
            # 角速度控制：将朝向误差 rotvec 比例映射为角速度，并进行限幅。
            if self.enable_angular and rotvec_error_norm > self.angular_deadband:
                angular_raw = self.angular_gain * rotvec_error
                angular_raw_norm = float(np.linalg.norm(angular_raw))
                angular_cmd, angular_limited_norm = self._clip_vector_norm(
                    angular_raw,
                    self.angular_speed_max,
                )
            # 同样施加角加速度斜率限制，减小控制突变。
            angular_cmd = self._slew_limit_vector(
                angular_cmd,
                self._last_angular_cmd,
                self.angular_accel_max * servo_dt_sec,
            )
            self._last_linear_cmd = linear_cmd.copy()
            self._last_angular_cmd = angular_cmd.copy()

            publish_motion_components(
                self.cmd_pub,
                cmd_frame=self.cmd_frame,
                linear_cmd=linear_cmd,
                angular_cmd=angular_cmd,
                stamp=pose_stamp,
            )
            self.profile_logger.debug(
                (
                    "servo_cmd: cmd_age_ms=%.1f timeout_ms=%.1f dt=%.4f "
                    "theta_phi_rate=(%.6f, %.6f) |vt|=%.6f |vr|=%.6f "
                    "radial_error=%.6f radial_p=%.6f radial_i=%.6f radial_integral=%.6f "
                    "radial_raw=%.6f radial_cmd=%.6f radial_limited=%s radial_i_accept=%s "
                    "linear_raw=%.6f linear_limited=%.6f linear_out=%.6f "
                    "rotvec_error=%.6f angular_raw=%.6f angular_limited=%.6f angular_out=%.6f"
                ),
                float(cmd_age * 1000.0),
                float(effective_timeout_sec * 1000.0),
                float(servo_dt_sec),
                float(theta_rate),
                float(phi_rate),
                float(np.linalg.norm(v_t)),
                float(np.linalg.norm(v_r)),
                float(radial_error),
                float(radial_p_term),
                float(radial_i_term),
                float(self._radial_integral_error),
                float(radial_speed_raw),
                float(radial_speed_cmd),
                bool(radial_limited),
                bool(radial_integral_accepted),
                float(linear_raw_norm),
                float(linear_limited_norm),
                float(np.linalg.norm(linear_cmd)),
                float(rotvec_error_norm),
                float(angular_raw_norm),
                float(angular_limited_norm),
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
            # 任意异常都 fail-safe 到零速，保护执行侧。
            publish_zero_twist(self.cmd_pub, cmd_frame=self.cmd_frame, stamp=pose_stamp)
            self._reset_slew_state()
            self.stats.servo_exception += 1
        self._maybe_log_status()


def build_argparser():
    parser = argparse.ArgumentParser(
        description="Lightweight InfoFlow servo runtime for execution-side hosts.",
    )
    parser.add_argument("-c", "--config", type=str, default="config/rtabmap_config.yaml", help="YAML config file path")
    parser.add_argument("--spherical_cmd_topic", type=str, default="/omnimap/spherical_cmd")
    parser.add_argument("--cmd_topic", type=str, default="/servo_server/delta_twist_camera")
    parser.add_argument("--cmd_frame", type=str, default="base_link")
    parser.add_argument("--world_frame", type=str, default="base_link")
    parser.add_argument("--camera_frame", type=str, default="cam_1_color_optical_frame")
    parser.add_argument("--servo_hz", type=float, default=None)
    parser.add_argument("--spherical_cmd_timeout_sec", type=float, default=None)
    parser.add_argument("--adaptive_cmd_timeout_scale", type=float, default=1.5)
    parser.add_argument("--adaptive_cmd_timeout_cap_sec", type=float, default=0.5)
    parser.add_argument("--pose_stale_timeout_sec", type=float, default=None)
    parser.add_argument("--linear_vel_max", type=float, default=None)
    parser.add_argument("--angular_speed_max", type=float, default=None)
    parser.add_argument("--radial_gain", type=float, default=None)
    parser.add_argument("--radial_i_gain", type=float, default=None)
    parser.add_argument("--radial_deadband", type=float, default=None)
    parser.add_argument("--radial_integral_limit", type=float, default=None)
    parser.add_argument("--radial_vel_max", type=float, default=None)
    parser.add_argument("--angular_gain", type=float, default=None)
    parser.add_argument("--angular_deadband", type=float, default=None)
    parser.add_argument("--linear_accel_max", type=float, default=None)
    parser.add_argument("--angular_accel_max", type=float, default=None)
    parser.add_argument("--enable_angular", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("-o", "--output", type=str, default=f"replica/output/{time.strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--log_level", type=str, default="INFO")
    parser.add_argument("--log_file", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--status_log_interval_sec", type=float, default=1.0)
    return parser


if __name__ == "__main__":
    parser = build_argparser()
    args = parser.parse_args()
    set_nofile_limit()
    os.makedirs(args.output, exist_ok=True)
    log_file = os.path.join(args.output, "servo_runtime.log") if bool(args.log_file) else None
    configure_servo_logging(level=args.log_level, log_file=log_file, force=True)

    mc = load_servo_motion_config(args.config)

    def _resolve(cli_val, yaml_val):
        return cli_val if cli_val is not None else yaml_val

    args.servo_hz = _resolve(args.servo_hz, mc["servo_hz"])
    args.spherical_cmd_timeout_sec = _resolve(args.spherical_cmd_timeout_sec, mc["spherical_cmd_timeout_sec"])
    args.pose_stale_timeout_sec = _resolve(args.pose_stale_timeout_sec, mc["pose_stale_timeout_sec"])
    args.linear_vel_max = _resolve(args.linear_vel_max, mc["linear_vel_max"])
    args.angular_speed_max = _resolve(args.angular_speed_max, mc["angular_speed_max"])
    args.radial_gain = _resolve(args.radial_gain, mc["radial_gain"])
    args.radial_i_gain = _resolve(args.radial_i_gain, mc["radial_i_gain"])
    args.radial_deadband = _resolve(args.radial_deadband, mc["radial_deadband"])
    args.radial_integral_limit = _resolve(args.radial_integral_limit, mc["radial_integral_limit"])
    args.radial_vel_max = _resolve(args.radial_vel_max, mc["radial_vel_max"])
    args.angular_gain = _resolve(args.angular_gain, mc["angular_gain"])
    args.angular_deadband = _resolve(args.angular_deadband, mc["angular_deadband"])
    args.linear_accel_max = _resolve(args.linear_accel_max, mc["linear_accel_max"])
    args.angular_accel_max = _resolve(args.angular_accel_max, mc["angular_accel_max"])
    args.enable_angular = _resolve(args.enable_angular, mc["enable_angular"])

    node = InfoFlowServoRuntime(args)
    rospy.spin()
