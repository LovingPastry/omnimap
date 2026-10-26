from __future__ import annotations

import argparse
import copy
import os
import time
import threading
from dataclasses import dataclass
from typing import Optional

import numpy as np
import rospy
import tf2_ros
from geometry_msgs.msg import TwistStamped
from omnimap_msgs.msg import ServoTwistTrace, KinematicStatus
from omnimap_msgs.srv import EvaluateKinematicModel
from scipy.spatial.transform import Rotation as R
from kinematic_projection import camera_rate_model, unclipped_preview_rate_model
from kinematic_constraint import KinematicConstraintGate
from bounded_query import SingleFlightQuery
from preview_joint_state import PreviewJointState
from projection_readiness import ProjectionReadinessGate
from sensor_msgs.msg import JointState

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


@dataclass(frozen=True)
class ServoCandidate:
    linear: np.ndarray
    angular: np.ndarray
    radial_integral: float
    rate_model: tuple
    rate_limits: dict
    details: dict


@dataclass(frozen=True)
class PreviewEvaluation:
    constraint: object
    command: object
    trace: object
    pose: np.ndarray
    received_wall: float


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
    servo_zero_frame_mismatch: int = 0
    servo_exception: int = 0


class InfoFlowServoRuntime:
    def __init__(self, args):
        self._control_lock = threading.RLock()
        self._projection_status_lock = threading.Lock()
        self._projection_health_epoch = 0
        self._control_generation = 0
        self._preview_queries = SingleFlightQuery()
        self._cycle_projection_queries = SingleFlightQuery()
        self.cycle_projection_budget_sec = float(args.kinematic_cycle_projection_budget_sec)
        if not np.isfinite(self.cycle_projection_budget_sec) or self.cycle_projection_budget_sec <= 0:
            raise ValueError("kinematic_cycle_projection_budget_sec must be finite and positive")
        self._preview_joints = PreviewJointState()
        self.preview_mode = args.kinematic_preview_mode
        if self.preview_mode not in ("off", "observe"):
            raise ValueError("invalid kinematic_preview_mode")
        self.preview_hz = float(args.kinematic_preview_hz)
        if not np.isfinite(self.preview_hz) or self.preview_hz <= 0:
            raise ValueError("kinematic_preview_hz must be finite and positive")
        self.latest_raw_cmd = None
        self.latest_preview_constraint = None
        self.latest_preview_evaluation = None
        self.latest_cycle_proposal = None
        self.preview_query = None
        self.preview_timer = None
        self.raw_cmd_sub = None
        self.preview_timeout_sec = float(args.kinematic_preview_timeout_sec)
        if not np.isfinite(self.preview_timeout_sec) or self.preview_timeout_sec <= 0:
            raise ValueError("kinematic_preview_timeout_sec must be finite and positive")
        rospy.init_node("info_flow_servo_runtime", anonymous=True)
        spherical_command_msg = import_spherical_command_msg()

        self.main_logger = get_section_logger("entry.infoflow_servo_runtime", "main")
        self.planner_logger = get_section_logger("planner.infoflow_servo_runtime", "planner")
        self.profile_logger = get_section_logger("profile.infoflow_servo_runtime", "profile")
        self.world_frame = str(args.world_frame)
        self.camera_frame = str(args.camera_frame)
        self.cmd_frame = str(args.cmd_frame)
        self.cmd_topic = str(args.cmd_topic)
        if (self.preview_mode == "observe" and
                rospy.resolve_name(args.spherical_cmd_raw_topic) == rospy.resolve_name(args.spherical_cmd_topic)):
            raise ValueError("preview requires distinct raw and final command topics")
        if rospy.resolve_name(args.twist_trace_topic) == rospy.resolve_name(self.cmd_topic):
            raise ValueError("twist_trace_topic and cmd_topic must differ")
        if (not self.world_frame.lstrip("/") or
                self.cmd_frame.lstrip("/") != self.world_frame.lstrip("/")):
            raise ValueError("servo computes Twist in world_frame; cmd_frame must match it")
        self.servo_hz = float(args.servo_hz)
        self.pose_stale_timeout_sec = float(args.pose_stale_timeout_sec)
        self.spherical_cmd_timeout_sec = float(args.spherical_cmd_timeout_sec)
        self.adaptive_cmd_timeout_scale = max(1.0, float(args.adaptive_cmd_timeout_scale))
        self.adaptive_cmd_timeout_cap_sec = max(
            self.spherical_cmd_timeout_sec,
            float(args.adaptive_cmd_timeout_cap_sec),
        )
        self.linear_vel_max = float(args.linear_vel_max)
        self.linear_deadband = max(0.0, float(args.linear_deadband))
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

        self.projection_readiness = ProjectionReadinessGate(
            source_frame=self.world_frame, timeout_sec=self.pose_stale_timeout_sec,
            recovery_sec=args.kinematic_projection_recovery_sec)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.cmd_pub = rospy.Publisher(args.cmd_topic, TwistStamped, queue_size=1)
        self.twist_trace_pub = rospy.Publisher(args.twist_trace_topic, ServoTwistTrace, queue_size=10)
        self.latest_cmd: Optional[CachedSphericalCommand] = None
        self._last_cmd_receipt_wall: Optional[float] = None
        self._cmd_interval_ema_sec: Optional[float] = None
        self._last_linear_cmd = np.zeros(3, dtype=np.float64)
        self._last_angular_cmd = np.zeros(3, dtype=np.float64)
        self._last_servo_wall: Optional[float] = None
        self._radial_integral_error = 0.0

        self.stats = ServoStats()
        self._last_status_wall = float(time.monotonic())
        self._last_status_steps = 0
        self._last_status_nonzero = 0
        self._throttle_last = {}

        self.cmd_sub = rospy.Subscriber(
            args.spherical_cmd_topic,
            spherical_command_msg,
            self.spherical_cmd_callback,
            queue_size=1,
        )
        self.preview_status_sub = rospy.Subscriber(
            args.kinematic_status_topic, KinematicStatus, self.preview_status_callback, queue_size=1)
        self.preview_joint_sub = rospy.Subscriber(
            args.kinematic_joint_topic, JointState, self.preview_joint_callback, queue_size=1)
        self.timer = rospy.Timer(
            rospy.Duration(1.0 / max(self.servo_hz, 1e-6)),
            self._servo_timer_callback,
        )

        if self.preview_mode == "observe":
            self.preview_query = rospy.ServiceProxy(args.kinematic_preview_service, EvaluateKinematicModel)
            self.raw_cmd_sub = rospy.Subscriber(args.spherical_cmd_raw_topic, spherical_command_msg,
                                               self.raw_spherical_cmd_callback, queue_size=1)
            self.preview_timer = rospy.Timer(rospy.Duration(1./self.preview_hz), self._preview_timer_callback)

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
                "linear_accel_max=%.4f angular_accel_max=%.4f linear_deadband=%.4f"
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
            float(self.linear_deadband),
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
            # <=0 表示禁用该标量限幅（保持原值直通）。
            return value, False
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

    def _make_twist_trace(self, msg, *, cmd, reason, rate_model=None, rate_limits=None):
        trace = ServoTwistTrace()
        trace.header.stamp = msg.header.stamp
        trace.header.frame_id = msg.header.frame_id
        trace.planner_tick = int(cmd.planner_tick) if cmd is not None else 0
        trace.model_version = int(cmd.model_version) if cmd is not None else 0
        trace.command_stamp = cmd.header.stamp if cmd is not None else rospy.Time()
        trace.command = msg
        trace.reason = reason
        trace.rate_model_reason = "no_motion_model"
        if rate_model is not None and cmd is not None:
            valid, model_reason, matrix, bias = rate_model
            trace.rate_model_valid = bool(valid)
            trace.rate_model_reason = model_reason
            trace.reference_radius = float(cmd.reference_radius)
            trace.reference_scene_center = list(cmd.reference_scene_center)
            trace.spherical_rates = [float(cmd.theta_rate), float(cmd.phi_rate)]
            trace.camera_rate_matrix = matrix.reshape(-1).tolist()
            trace.camera_bias = bias.tolist()
            if rate_limits is not None:
                trace.linear_limits_valid = True
                trace.previous_linear = rate_limits["previous"].tolist()
                trace.linear_speed_limit = rate_limits["speed"]
                trace.linear_step_limit = rate_limits["step"]
                trace.linear_deadband = rate_limits["deadband"]
                trace.servo_dt_sec = rate_limits["dt"]
        return trace

    def raw_spherical_cmd_callback(self, msg):
        self.latest_raw_cmd = CachedSphericalCommand(copy.deepcopy(msg), time.monotonic())

    def _preview_command_pair(self):
        raw, final = self.latest_raw_cmd, self.latest_cmd
        if raw is None or final is None:
            raise ValueError("preview raw/final command missing")
        raw, final = copy.deepcopy(raw), copy.deepcopy(final)
        now = time.monotonic()
        timeout = self._effective_cmd_timeout_sec()
        if any(not 0 <= now-cache.receipt_wall_time <= timeout for cache in (raw, final)):
            raise ValueError("preview raw/final command stale")
        a, b = raw.msg, final.msg
        if (a.header.stamp != b.header.stamp or a.header.frame_id != b.header.frame_id
                or a.planner_tick != b.planner_tick
                or a.model_version != b.model_version or a.reference_radius != b.reference_radius
                or list(a.reference_scene_center) != list(b.reference_scene_center)):
            raise ValueError("preview raw/final command mismatch")
        if a.should_stop:
            raise ValueError("preview policy requests stop")
        return raw, final

    def _preview_timer_callback(self, _event):
        if self.preview_mode != "observe":
            return
        self.latest_preview_constraint = None
        self.latest_preview_evaluation = None
        try:
            with self._projection_status_lock:
                self._validate_projection_readiness()
                health_epoch = self._projection_health_epoch
            raw, final = self._preview_command_pair()
            pose, stamp = lookup_latest_pose_from_tf(
                self.tf_buffer, world_frame=self.world_frame, camera_frame=self.camera_frame, timeout_sec=0.)
            # A proposed interval only: this callback must not advance the
            # actual Servo clock or claim a future measured execution dt.
            proposed_dt = 1./self.servo_hz
            evaluation = self.evaluate_preview(raw.msg, pose, stamp, proposed_dt, self.preview_query,
                                               return_evaluation=True)
            response = evaluation.constraint
            new_raw, new_final = self._preview_command_pair()
            if new_raw.msg != raw.msg or new_final.msg != final.msg:
                raise ValueError("preview command changed during query")
            with self._projection_status_lock:
                self._validate_projection_readiness()
                if self._projection_health_epoch != health_epoch:
                    raise ValueError("projection health epoch changed during work")
                self.latest_preview_constraint = response
                self.latest_preview_evaluation = evaluation
            self.profile_logger.info(
                "kinematic_preview: mode=observe tick=%d valid=True reason=%s threshold=%s "
                "dt_source=nominal dt=%.9g sigma_min=%.9g",
                raw.msg.planner_tick, response.reason, response.threshold_configured,
                proposed_dt, response.sigma_min)
        except Exception as exc:
            self.profile_logger.info("kinematic_preview: mode=observe valid=False reason=%s", exc)

    def preview_status_callback(self, msg):
        now, wall = rospy.Time.now().to_sec(), time.monotonic()
        with self._projection_status_lock:
            self.projection_readiness.update(msg, ros_now=now, wall_now=wall)
            if not self.projection_readiness.decision(ros_now=now, wall_now=wall).ready:
                self._projection_health_epoch += 1
                self.latest_cycle_proposal = None
                self.latest_preview_constraint = None
                self.latest_preview_evaluation = None

    def _validate_projection_readiness(self):
        decision = self.projection_readiness.decision(
            ros_now=rospy.Time.now().to_sec(), wall_now=time.monotonic())
        if not decision.ready:
            raise ValueError(decision.reason)

    def preview_joint_callback(self, msg):
        self._preview_joints.update(msg, time.monotonic(), ros_now=rospy.Time.now().to_sec())

    def evaluate_preview(self, cmd, current_c2w, pose_stamp, servo_dt_sec, query, *, return_evaluation=False):
        """Read-only candidate query; never commits controller state or motion.

        query is an EvaluateKinematicModel service proxy. Calling this method
        does not authorize use of its response and is not on the Servo timer.
        """
        cmd = copy.deepcopy(cmd)
        current_c2w = np.array(current_c2w, dtype=float, copy=True)
        now = rospy.Time.now().to_sec()
        for stamp in (cmd.header.stamp, pose_stamp):
            age = now-stamp.to_sec()
            if stamp.is_zero() or age < -.05 or age > self.pose_stale_timeout_sec:
                raise ValueError("preview source is stale")
        if cmd.header.frame_id.lstrip("/") != self.world_frame.lstrip("/"):
            raise ValueError("preview command frame mismatch")
        with self._control_lock:
            state_before = (self._radial_integral_error, self._last_linear_cmd.copy(),
                            self._last_angular_cmd.copy(), self._control_generation)
            candidate = self._compute_servo_candidate(cmd, current_c2w, servo_dt_sec)
        msg = TwistStamped()
        msg.header.stamp, msg.header.frame_id = pose_stamp, self.cmd_frame
        preview_model = unclipped_preview_rate_model(candidate.rate_model, [cmd.theta_rate, cmd.phi_rate])
        _, _, matrix, bias = preview_model
        mathematical = matrix @ np.asarray([cmd.theta_rate, cmd.phi_rate]) + bias
        msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z = mathematical[:3]
        msg.twist.angular.x, msg.twist.angular.y, msg.twist.angular.z = mathematical[3:]
        trace = self._make_twist_trace(msg, cmd=cmd, reason="preview",
                                      rate_model=preview_model, rate_limits=candidate.rate_limits)
        if not trace.rate_model_valid:
            raise ValueError("preview model unavailable:" + trace.rate_model_reason)
        started = time.monotonic()
        deadline = min(self.preview_timeout_sec, self.pose_stale_timeout_sec)
        response = self._preview_queries.call(query, trace, deadline).constraint
        if time.monotonic()-started > deadline:
            raise ValueError("preview response deadline exceeded")
        self._validate_preview_response(trace, response)
        latest_pose, latest_stamp = lookup_latest_pose_from_tf(
            self.tf_buffer, world_frame=self.world_frame, camera_frame=self.camera_frame, timeout_sec=0.)
        latest_age = rospy.Time.now().to_sec()-latest_stamp.to_sec()
        if latest_stamp.is_zero() or latest_age < -.05 or latest_age > self.pose_stale_timeout_sec:
            raise ValueError("preview latest pose is stale")
        if (np.asarray(latest_pose).shape != (4, 4)
                or not np.all(np.isfinite(latest_pose))
                or not np.allclose(latest_pose, current_c2w, rtol=0, atol=1e-9)):
            raise ValueError("preview pose changed during query")
        self._preview_joints.validate(response, ros_now=rospy.Time.now().to_sec(),
                                      wall_now=time.monotonic(), timeout=self.pose_stale_timeout_sec)
        with self._control_lock:
            if (self._control_generation != state_before[3]
                    or self._radial_integral_error != state_before[0]
                    or not np.array_equal(self._last_linear_cmd, state_before[1])
                    or not np.array_equal(self._last_angular_cmd, state_before[2])):
                raise ValueError("preview controller state changed during query")
        if time.monotonic()-started > deadline:
            raise ValueError("preview response deadline exceeded")
        if return_evaluation:
            return PreviewEvaluation(copy.deepcopy(response), cmd, trace, current_c2w, time.monotonic())
        return response

    def revalidate_cycle_preview(self, evaluation, raw_cmd, current_pose, servo_dt_sec):
        """Rebuild current-cycle limits and solve a non-actuating projection.

        Caller holds the control lock or owns an isolated controller snapshot.
        A changed affine map needs a new model
        query; changing only dt/previous linear limits is handled by re-solving.
        """
        self._validate_projection_readiness()
        if raw_cmd != evaluation.command or raw_cmd.should_stop:
            raise ValueError("preview cycle command mismatch or policy stop")
        if not np.allclose(current_pose, evaluation.pose, rtol=0, atol=1e-9):
            raise ValueError("preview cycle pose changed")
        self._preview_joints.validate(evaluation.constraint, ros_now=rospy.Time.now().to_sec(),
                                      wall_now=time.monotonic(), timeout=self.pose_stale_timeout_sec)
        current = self._compute_servo_candidate(raw_cmd, current_pose, servo_dt_sec)
        valid, reason, matrix, bias = unclipped_preview_rate_model(
            current.rate_model, [raw_cmd.theta_rate, raw_cmd.phi_rate])
        if (not valid or not np.allclose(matrix.reshape(-1), evaluation.trace.camera_rate_matrix,
                                        rtol=0, atol=1e-9)
                or not np.allclose(bias, evaluation.trace.camera_bias, rtol=0, atol=1e-9)):
            raise ValueError("preview cycle affine model changed:" + reason)
        msg = copy.deepcopy(evaluation.constraint)
        msg.servo_dt_sec = servo_dt_sec
        msg.previous_linear = current.rate_limits["previous"].tolist()
        msg.linear_speed_limit = current.rate_limits["speed"]
        msg.linear_step_limit = current.rate_limits["step"]
        msg.linear_deadband = current.rate_limits["deadband"]
        gate = KinematicConstraintGate(source_frame=self.world_frame,
                                       timeout_sec=self.pose_stale_timeout_sec)
        gate.remember_preview_command(raw_cmd)
        gate.update(msg, wall_now=evaluation.received_wall)
        proposal = gate.propose(rates=[raw_cmd.theta_rate, raw_cmd.phi_rate],
            reference_radius=raw_cmd.reference_radius, reference_scene_center=raw_cmd.reference_scene_center,
            model_version=raw_cmd.model_version, should_stop=raw_cmd.should_stop,
            state_allows_motion=True, ros_now=rospy.Time.now().to_sec(), wall_now=time.monotonic())
        if proposal.valid:
            projected = copy.deepcopy(raw_cmd)
            projected.theta_rate, projected.phi_rate = proposal.rates
            candidate = self._compute_servo_candidate(projected, current_pose, servo_dt_sec)
            predicted = matrix @ np.asarray(proposal.rates) + bias
            if not np.allclose(np.r_[candidate.linear, candidate.angular], predicted, rtol=0, atol=1e-9):
                raise ValueError("preview projected output changed by runtime limits")
        return proposal

    def _observe_cycle_projection(self, cmd, pose, servo_dt_sec):
        if self.preview_mode != "observe":
            return
        self.latest_cycle_proposal = None
        try:
            with self._projection_status_lock:
                self._validate_projection_readiness()
                health_epoch = self._projection_health_epoch
            evaluation = self.latest_preview_evaluation
            if evaluation is None:
                raise ValueError("preview cycle model missing")
            raw, final = self._preview_command_pair()
            if final.msg != cmd:
                raise ValueError("preview cycle final command changed")
            # Numerical optimization must not hold up the control timer. The
            # worker owns a controller snapshot and cannot commit runtime state.
            snapshot = copy.copy(self)
            snapshot._last_linear_cmd = self._last_linear_cmd.copy()
            snapshot._last_angular_cmd = self._last_angular_cmd.copy()
            budget = min(self.cycle_projection_budget_sec, .25/max(self.servo_hz, 1e-6))
            proposal = self._cycle_projection_queries.call(
                lambda payload: snapshot.revalidate_cycle_preview(*payload),
                (evaluation, raw.msg, np.array(pose, copy=True), servo_dt_sec), budget)
            with self._projection_status_lock:
                self._validate_projection_readiness()
                if self._projection_health_epoch != health_epoch:
                    raise ValueError("projection health epoch changed during work")
                self.latest_cycle_proposal = proposal
            self.profile_logger.info(
                "kinematic_cycle_projection: mode=observe tick=%d valid=%s reason=%s "
                "dt_source=actual dt=%.9g proposed=[%.9g %.9g]",
                cmd.planner_tick, proposal.valid, proposal.reason, servo_dt_sec, *proposal.rates)
        except Exception as exc:
            self.profile_logger.info("kinematic_cycle_projection: mode=observe valid=False reason=%s", exc)

    def _validate_preview_response(self, trace, response):
        """Associate an advisory response; this does not authorize execution."""
        if not response.valid or not response.reason.startswith("advisory_preview:"):
            raise ValueError("preview response invalid:" + response.reason)
        if (response.header.frame_id.lstrip("/") != self.world_frame.lstrip("/")
                or response.source_frame.lstrip("/") != self.world_frame.lstrip("/")
                or response.task_frame != "tool0"
                or response.planner_tick != trace.planner_tick
                or response.model_version != trace.model_version
                or response.command_stamp != trace.command_stamp
                or response.twist_stamp != trace.command.header.stamp
                or response.twist_seq != trace.command.header.seq):
            raise ValueError("preview response provenance mismatch")
        now = rospy.Time.now().to_sec()
        for stamp in (response.header.stamp, response.joint_stamp, response.twist_stamp,
                      response.command_stamp):
            age = now-stamp.to_sec()
            if stamp.is_zero() or age < -.05 or age > self.pose_stale_timeout_sec:
                raise ValueError("preview response is stale")
        if abs((response.joint_stamp-response.twist_stamp).to_sec()) > self.pose_stale_timeout_sec:
            raise ValueError("preview response joint skew")
        scales = np.asarray(response.rate_scales)
        if scales.shape != (2,) or not np.all(np.isfinite(scales)) or np.any(scales <= 0):
            raise ValueError("preview response rate scales invalid")
        expected_matrix = np.asarray(trace.camera_rate_matrix).reshape(6, 2)[:3]*scales
        pairs = ((response.reference_radius, trace.reference_radius),
                 (response.reference_scene_center, trace.reference_scene_center),
                 (response.linearization_rates, trace.spherical_rates),
                 (response.camera_linear_matrix, expected_matrix.reshape(-1)),
                 (response.camera_linear_bias, trace.camera_bias[:3]),
                 (response.previous_linear, trace.previous_linear),
                 (response.servo_dt_sec, trace.servo_dt_sec),
                 (response.linear_speed_limit, trace.linear_speed_limit),
                 (response.linear_step_limit, trace.linear_step_limit),
                 (response.linear_deadband, trace.linear_deadband))
        for actual, expected in pairs:
            actual, expected = np.asarray(actual), np.asarray(expected)
            if (actual.shape != expected.shape or not np.all(np.isfinite(actual))
                    or not np.allclose(actual, expected, rtol=0, atol=1e-12)):
                raise ValueError("preview response model mismatch")

    def _publish_twist(self, *, cmd=None, stamp=None, linear_cmd=None,
                       angular_cmd=None, reason, rate_model=None, rate_limits=None):
        """Log the exact camera-reference Twist sent to the bridge, including zeros."""
        if linear_cmd is None:
            msg = publish_zero_twist(self.cmd_pub, cmd_frame=self.cmd_frame, stamp=stamp)
        else:
            msg = publish_motion_components(
                self.cmd_pub, cmd_frame=self.cmd_frame, stamp=stamp,
                linear_cmd=linear_cmd, angular_cmd=angular_cmd,
            )
        linear, angular = msg.twist.linear, msg.twist.angular
        trace = self._make_twist_trace(msg, cmd=cmd, reason=reason,
                                       rate_model=rate_model, rate_limits=rate_limits)
        self.twist_trace_pub.publish(trace)
        self.profile_logger.info(
            "servo_twist: tick=%d model_v=%d command_stamp=%.9f twist_stamp=%.9f "
            "topic=%s frame=%s reference=%s linear=[%.9g %.9g %.9g] "
            "angular=[%.9g %.9g %.9g] reason=%s",
            int(cmd.planner_tick) if cmd is not None else 0,
            int(cmd.model_version) if cmd is not None else 0,
            cmd.header.stamp.to_sec() if cmd is not None else 0.0,
            msg.header.stamp.to_sec(), self.cmd_topic, msg.header.frame_id,
            self.camera_frame, linear.x, linear.y, linear.z,
            angular.x, angular.y, angular.z, reason,
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
                "Servo 状态：servo_hz=%.1f cmd_hz=%.1f "
                "conn(sub=%d pub=%d) 累计(rx=%d steps=%d nonzero=%d missing_cmd=%d cmd_stale=%d pose_stale=%d tf_fail=%d policy_stop=%d frame_mismatch=%d exc=%d)"
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
            int(stats.servo_zero_frame_mismatch),
            int(stats.servo_exception),
        )

    def _compute_servo_candidate(self, cmd, current_c2w, servo_dt_sec):
        """Calculate one frozen cycle without changing PI, slew or publishers."""
        if not np.isfinite(servo_dt_sec) or servo_dt_sec <= 0:
            raise ValueError("invalid Servo candidate period")
        radial_integral = float(self._radial_integral_error)
        current_position = np.asarray(current_c2w[:3, 3], dtype=np.float64)
        reference_scene_center = np.asarray(
            cmd.reference_scene_center,
            dtype=np.float64,
        ).reshape(3)
        radius, theta, phi = position_to_spherical(current_position, reference_scene_center)
        e_theta, e_phi, n_hat = local_frame_from_theta_phi(theta, phi)
        theta_rate = float(cmd.theta_rate)
        phi_rate = float(cmd.phi_rate)
        if not np.all(np.isfinite([theta_rate, phi_rate, cmd.reference_radius,
                                   *reference_scene_center])):
            raise ValueError("nonfinite spherical command")
        # 切向速度：沿球面切平面的角速度分量。
        v_t = radius * (theta_rate * e_theta + phi_rate * e_phi)

        reference_radius = float(cmd.reference_radius)
        radial_error = float(reference_radius - radius)
        # 径向 PI：负责把当前半径收敛到参考半径，带抗积分饱和与速度限幅。
        radial_p_term = self.radial_gain * radial_error
        radial_i_term = self.radial_i_gain * radial_integral
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
                radial_integral = float(
                    np.clip(
                        radial_integral + radial_error * servo_dt_sec,
                        -self.radial_integral_limit,
                        self.radial_integral_limit,
                    )
                )
                radial_integral_accepted = True
            radial_i_term = self.radial_i_gain * radial_integral
            radial_speed_raw = radial_p_term + radial_i_term
            radial_speed_cmd, radial_limited = self._clip_scalar(
                radial_speed_raw,
                self.radial_vel_max,
            )
            v_r = radial_speed_cmd * n_hat
        else:
            # 在 deadband 内清零积分，避免微小抖动长期累积。
            radial_integral = 0.0
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
        # 线速度死区：当输出线速度足够小则直接清零，避免机械臂长期“慢爬”。
        linear_cmd_norm = float(np.linalg.norm(linear_cmd))
        if linear_cmd_norm < self.linear_deadband:
            linear_cmd = np.zeros(3, dtype=np.float64)
            linear_cmd_norm = 0.0

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
        rate_model = camera_rate_model(
            radius=radius, e_theta=e_theta, e_phi=e_phi,
            rates=[theta_rate, phi_rate], radial_velocity=v_r,
            linear_command=linear_cmd, angular_command=angular_cmd,
            previous_linear=self._last_linear_cmd,
            linear_speed_limit=self.linear_vel_max,
            linear_step_limit=self.linear_accel_max * servo_dt_sec,
            linear_deadband=self.linear_deadband,
        )
        rate_limits = dict(previous=self._last_linear_cmd.copy(), speed=max(0.0, self.linear_vel_max),
                           step=self.linear_accel_max * servo_dt_sec, deadband=self.linear_deadband,
                           dt=servo_dt_sec)
        return ServoCandidate(linear_cmd, angular_cmd, radial_integral,
                              rate_model, rate_limits, {
            'reference_scene_center': reference_scene_center,
            'reference_radius': reference_radius,
            'radius': radius,
            'radial_error': radial_error,
            'theta_rate': theta_rate,
            'phi_rate': phi_rate,
            'v_t': v_t,
            'v_r': v_r,
            'radial_p_term': radial_p_term,
            'radial_i_term': radial_i_term,
            'radial_speed_raw': radial_speed_raw,
            'radial_speed_cmd': radial_speed_cmd,
            'radial_limited': radial_limited,
            'radial_integral_accepted': radial_integral_accepted,
            'linear_raw_norm': linear_raw_norm,
            'linear_limited_norm': linear_limited_norm,
            'linear_cmd_norm': linear_cmd_norm,
            'rotvec_error_norm': rotvec_error_norm,
            'angular_raw_norm': angular_raw_norm,
            'angular_limited_norm': angular_limited_norm,
        })

    def _servo_timer_callback(self, _event) -> None:
        with self._control_lock:
            self._control_generation += 1
            self._servo_step()

    def _servo_step(self) -> None:
        # 周期伺服主循环：
        # 1) 读取最新 planner 命令与本机位姿；
        # 2) 通过多重安全门（缺命令/过期/TF失败/位姿过期/策略要求停止）；
        # 3) 将球坐标速度指令转换为 Twist 并发布。
        self.stats.servo_steps += 1
        if self.preview_mode == "observe":
            self.latest_cycle_proposal = None

        cmd_cache = self.latest_cmd
        if cmd_cache is None:
            # 尚未收到任何规划命令：主动发布零速，保证执行侧静止安全。
            self._publish_twist(reason="missing_command")
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
            self._publish_twist(cmd=cmd, reason="command_stale")
            self._reset_slew_state()
            self.stats.servo_zero_cmd_stale += 1
            self._maybe_log_status()
            return

        if bool(cmd.should_stop):
            diagnostic_dt = self._servo_dt_sec() if self.preview_mode == "observe" else None
            self._publish_twist(cmd=cmd, reason=str(cmd.stop_reason))
            self._reset_slew_state()
            self.stats.servo_zero_policy_stop += 1
            if self.preview_mode == "observe":
                try:
                    pose, stamp = lookup_latest_pose_from_tf(
                        self.tf_buffer, world_frame=self.world_frame, camera_frame=self.camera_frame,
                        timeout_sec=0.)
                    age = rospy.Time.now().to_sec()-stamp.to_sec()
                    if stamp.is_zero() or not -.05 <= age <= self.pose_stale_timeout_sec:
                        raise ValueError("stopped cycle pose stale")
                    self._observe_cycle_projection(cmd, pose, diagnostic_dt)
                except Exception as exc:
                    self.latest_cycle_proposal = None
                    self.profile_logger.info("kinematic_cycle_projection: mode=observe valid=False reason=%s", exc)
            self._maybe_log_status()
            return
        if cmd.header.frame_id.lstrip("/") != self.world_frame.lstrip("/"):
            self._publish_twist(cmd=cmd, reason="command_frame_mismatch")
            self._reset_slew_state()
            self.stats.servo_zero_frame_mismatch += 1
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
            self._publish_twist(cmd=cmd, reason="tf_failure")
            self._reset_slew_state()
            self.stats.servo_zero_tf_fail += 1
            self._maybe_log_status()
            return

        pose_age = float((rospy.Time.now() - pose_stamp).to_sec())
        if pose_age > self.pose_stale_timeout_sec or pose_age < -0.05:
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
            self._publish_twist(cmd=cmd, stamp=pose_stamp, reason="pose_stale")
            self._reset_slew_state()
            self.stats.servo_zero_pose_stale += 1
            self._maybe_log_status()
            return

        try:
            servo_dt_sec = self._servo_dt_sec()
            self._observe_cycle_projection(cmd, current_c2w, servo_dt_sec)
            candidate = self._compute_servo_candidate(cmd, current_c2w, servo_dt_sec)
            linear_cmd, angular_cmd = candidate.linear, candidate.angular
            rate_model, rate_limits = candidate.rate_model, candidate.rate_limits
            self._radial_integral_error = candidate.radial_integral
            self._last_linear_cmd = linear_cmd.copy()
            self._last_angular_cmd = angular_cmd.copy()

            self._publish_twist(
                cmd=cmd,
                linear_cmd=linear_cmd,
                angular_cmd=angular_cmd,
                stamp=pose_stamp,
                reason="motion",
                rate_model=rate_model,
                rate_limits=rate_limits,
            )
            self._planner_log_throttle(
                "reference_sphere_servo",
                1.0,
                "info",
                "reference_sphere_servo: center=[%.3f %.3f %.3f] reference_radius=%.3fm current_radius=%.3fm radial_error=%.3fm",
                float(candidate.details['reference_scene_center'][0]),
                float(candidate.details['reference_scene_center'][1]),
                float(candidate.details['reference_scene_center'][2]),
                float(candidate.details['reference_radius']),
                float(candidate.details['radius']),
                float(candidate.details['radial_error']),
            )
            self.profile_logger.debug(
                (
                    "servo_cmd: cmd_age_ms=%.1f timeout_ms=%.1f dt=%.4f "
                    "theta_phi_rate=(%.6f, %.6f) |vt|=%.6f |vr|=%.6f "
                    "radial_error=%.6f radial_p=%.6f radial_i=%.6f radial_integral=%.6f "
                    "radial_raw=%.6f radial_cmd=%.6f radial_limited=%s radial_i_accept=%s "
                    "linear_raw=%.6f linear_limited=%.6f linear_out=%.6f "
                    "linear_deadband=%.6f "
                    "rotvec_error=%.6f angular_raw=%.6f angular_limited=%.6f angular_out=%.6f"
                ),
                float(cmd_age * 1000.0),
                float(effective_timeout_sec * 1000.0),
                float(servo_dt_sec),
                float(candidate.details['theta_rate']),
                float(candidate.details['phi_rate']),
                float(np.linalg.norm(candidate.details['v_t'])),
                float(np.linalg.norm(candidate.details['v_r'])),
                float(candidate.details['radial_error']),
                float(candidate.details['radial_p_term']),
                float(candidate.details['radial_i_term']),
                float(self._radial_integral_error),
                float(candidate.details['radial_speed_raw']),
                float(candidate.details['radial_speed_cmd']),
                bool(candidate.details['radial_limited']),
                bool(candidate.details['radial_integral_accepted']),
                float(candidate.details['linear_raw_norm']),
                float(candidate.details['linear_limited_norm']),
                float(candidate.details['linear_cmd_norm']),
                float(self.linear_deadband),
                float(candidate.details['rotvec_error_norm']),
                float(candidate.details['angular_raw_norm']),
                float(candidate.details['angular_limited_norm']),
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
            self._publish_twist(cmd=cmd, stamp=pose_stamp, reason=f"servo_exception:{exc}")
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
    parser.add_argument("--kinematic_status_topic", default="/omnimap/kinematic_status")
    parser.add_argument("--kinematic_projection_recovery_sec", type=float, default=0.1)
    parser.add_argument("--kinematic_cycle_projection_budget_sec", type=float, default=0.003)
    parser.add_argument("--kinematic_preview_mode", choices=("off", "observe"), default="off")
    parser.add_argument("--kinematic_preview_hz", type=float, default=2.)
    parser.add_argument("--kinematic_preview_service", default="/omnimap_kinematic_monitor/evaluate_model")
    parser.add_argument("--spherical_cmd_raw_topic", default="/omnimap/spherical_cmd_raw")
    parser.add_argument("--kinematic_joint_topic", default="/joint_states")
    parser.add_argument("--kinematic_preview_timeout_sec", type=float, default=0.1)
    parser.add_argument("--twist_trace_topic", type=str, default="/omnimap/servo_twist_trace")
    parser.add_argument("--cmd_frame", type=str, default="base_link")
    parser.add_argument("--world_frame", type=str, default="base_link")
    parser.add_argument("--camera_frame", type=str, default="cam_1_color_optical_frame")
    parser.add_argument("--servo_hz", type=float, default=None)
    parser.add_argument("--spherical_cmd_timeout_sec", type=float, default=None)
    parser.add_argument("--adaptive_cmd_timeout_scale", type=float, default=1.5)
    parser.add_argument("--adaptive_cmd_timeout_cap_sec", type=float, default=0.5)
    parser.add_argument("--pose_stale_timeout_sec", type=float, default=None)
    parser.add_argument("--linear_vel_max", type=float, default=None)
    parser.add_argument("--linear_deadband", type=float, default=None)
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
    args.linear_deadband = _resolve(args.linear_deadband, mc["linear_deadband"])
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
