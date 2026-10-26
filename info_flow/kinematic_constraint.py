"""Validate execution-side local models and compute non-actuating proposals.

This consumer deliberately has no enforce mode: model recovery, execution
revalidation and calibration remain to be integrated.
"""
from dataclasses import dataclass
import copy
import math

import numpy as np

from kinematic_projection import project_feasible_spherical_command


@dataclass(frozen=True)
class ProjectionProposal:
    valid: bool = False
    reason: str = "constraint_missing"
    rates: tuple = (0.0, 0.0)
    correction_norm: float = 0.0
    source_tick: int = 0
    acceleration_limits_complete: bool = False


class KinematicConstraintGate:
    """Caller must serialize update, remember_command, and propose with a lock."""

    def __init__(self, *, source_frame="base_link", task_frame="tool0",
                 timeout_sec=0.3, max_sample_skew_sec=0.1,
                 max_normalized_delta=0.25):
        self.source_frame, self.task_frame = source_frame.lstrip("/"), task_frame.lstrip("/")
        if not self.source_frame or not self.task_frame:
            raise ValueError("constraint frames must be nonempty")
        for value in (timeout_sec, max_sample_skew_sec, max_normalized_delta):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("constraint timeouts and local radius must be positive and finite")
        self.timeout_sec = timeout_sec
        self.max_sample_skew_sec = max_sample_skew_sec
        self.max_normalized_delta = max_normalized_delta
        self.latest, self.received_wall, self.command = None, 0.0, None

    def update(self, msg, *, wall_now):
        self.latest, self.received_wall = copy.deepcopy(msg), float(wall_now)

    def remember_command(self, msg):
        self._remember_command(msg, "execution")

    def remember_preview_command(self, msg):
        """Associate an explicitly non-actuating preview, never an execution model."""
        self._remember_command(msg, "preview")

    def _remember_command(self, msg, kind):
        # Keep source roles distinct: raw previews never authorize execution.

        self.command = dict(
            kind=kind, key=(int(msg.planner_tick), int(msg.model_version), float(msg.header.stamp.to_sec())),
            frame=msg.header.frame_id.lstrip("/"), stop=bool(msg.should_stop),
            rates=(float(msg.theta_rate), float(msg.phi_rate)),
            radius=float(msg.reference_radius), center=tuple(msg.reference_scene_center),
        )

    def propose(self, *, rates, reference_radius, reference_scene_center, model_version,
                should_stop, state_allows_motion, ros_now, wall_now):
        """Return a diagnostic suggestion; never mutate the command or inputs."""
        out = self._validate(
            rates, reference_radius, reference_scene_center, model_version,
            should_stop, state_allows_motion, ros_now, wall_now,
        )
        if out:
            return ProjectionProposal(reason=out)
        msg = self.latest
        scales = np.asarray(msg.rate_scales, dtype=float)
        original = np.asarray(rates, dtype=float) / scales
        origin = np.asarray(msg.linearization_rates, dtype=float) / scales
        if np.linalg.norm(original-origin) > self.max_normalized_delta + 1e-12:
            return ProjectionProposal(reason="constraint_candidate_outside_local_radius")
        try:
            joint_matrix = np.asarray(msg.joint_rate_matrix, dtype=float).reshape(6, 2)
            joint_bias = np.asarray(msg.joint_rate_bias, dtype=float)
            linear_matrix = np.vstack((joint_matrix, -joint_matrix))
            linear_bounds = np.r_[np.asarray(msg.joint_velocity_upper)-joint_bias,
                                  joint_bias-np.asarray(msg.joint_velocity_lower)]
            # One source-cycle Euler prediction, not a stopping-distance model.
            position_rows = np.asarray(msg.joint_position_bounded, dtype=bool)
            accel_rows = np.asarray(msg.joint_acceleration_bounded, dtype=bool)
            dt = msg.servo_dt_sec
            for selected, lower, upper in (
                    (position_rows,
                     (np.asarray(msg.joint_position_lower)-msg.joint_position)/dt,
                     (np.asarray(msg.joint_position_upper)-msg.joint_position)/dt),
                    (accel_rows,
                     np.asarray(msg.joint_velocity)+np.asarray(msg.joint_acceleration_lower)*dt,
                     np.asarray(msg.joint_velocity)+np.asarray(msg.joint_acceleration_upper)*dt)):
                matrix = joint_matrix[selected]
                linear_matrix = np.vstack((linear_matrix, matrix, -matrix))
                linear_bounds = np.r_[linear_bounds, (upper-joint_bias)[selected],
                                      (joint_bias-lower)[selected]]
            camera = np.asarray(msg.camera_linear_matrix, dtype=float).reshape(3, 2)
            bias = np.asarray(msg.camera_linear_bias, dtype=float)
            limits = []
            for limit, offset in ((msg.linear_speed_limit, bias),
                                  (msg.linear_step_limit, bias-np.asarray(msg.previous_linear))):
                if limit > 0:
                    limits.append((camera, offset, limit))
            if msg.linear_deadband > 0:
                # The exterior of the deadband ball is nonconvex. Keep the
                # convex supporting halfspace oriented by the source velocity.
                source_velocity = camera @ origin + bias
                speed = np.linalg.norm(source_velocity)
                if speed <= msg.linear_deadband:
                    raise ValueError("source velocity is inside deadband")
                normal = source_velocity / speed
                linear_matrix = np.vstack((linear_matrix, -normal @ camera))
                linear_bounds = np.r_[linear_bounds, normal @ bias-msg.linear_deadband]
            projected, norm = project_feasible_spherical_command(
                original, msg.a, msg.c, local_center=origin,
                local_radius=self.max_normalized_delta,
                linear_matrix=linear_matrix, linear_bounds=linear_bounds, norm_limits=limits)
        except ValueError as exc:
            return ProjectionProposal(reason="constraint_projection_unavailable:" + str(exc))
        if np.linalg.norm(projected-origin) > self.max_normalized_delta + 1e-9:
            return ProjectionProposal(reason="constraint_projection_outside_local_radius")
        return ProjectionProposal(True, "advisory_projection", tuple(projected * scales),
                                  norm, int(msg.planner_tick), bool(np.all(msg.joint_acceleration_bounded)))

    def _validate(self, rates, radius, center, model_version, should_stop, state_ok, ros_now, wall_now):
        if should_stop:
            return "constraint_policy_or_state_stop"
        if not state_ok:
            return "constraint_execution_state_unavailable"
        msg = self.latest
        if msg is None:
            return "constraint_missing"
        received_age = wall_now-self.received_wall
        stamps = [msg.header.stamp.to_sec(), msg.joint_stamp.to_sec(), msg.twist_stamp.to_sec()]
        if (not all(math.isfinite(v) for v in (ros_now, wall_now, received_age, *stamps))
                or any(stamp <= 0 for stamp in stamps)):
            return "constraint_invalid_time"
        ages = [ros_now-stamp for stamp in stamps]
        if received_age < 0 or min(ages) < -0.05:
            return "constraint_future"
        if max([received_age, *ages]) > self.timeout_sec:
            return "constraint_stale"
        if abs(stamps[1]-stamps[2]) > self.max_sample_skew_sec:
            return "constraint_sample_skew"
        if (msg.header.frame_id.lstrip("/") != self.source_frame
                or msg.source_frame.lstrip("/") != self.source_frame
                or msg.task_frame.lstrip("/") != self.task_frame):
            return "constraint_frame_mismatch"
        if not msg.valid:
            return "constraint_upstream_invalid:" + msg.reason
        if (self.command is not None and
                msg.reason.startswith("advisory_preview:") != (self.command["kind"] == "preview")):
            return "constraint_command_role_mismatch"
        if not msg.threshold_configured:
            return "constraint_threshold_unset"
        if (len(msg.joint_names) != 6 or len(set(msg.joint_names)) != 6
                or any(not name.strip() for name in msg.joint_names)):
            return "constraint_joint_names_invalid"
        scalars = (radius, msg.reference_radius, msg.sigma_min, msg.m_min, msg.alpha,
                   msg.c, msg.drift, msg.inverse_residual, msg.derivative_error,
                   msg.linear_speed_limit, msg.linear_step_limit, msg.linear_deadband, msg.servo_dt_sec)
        arrays = ((rates, (2,)), (center, (3,)), (msg.reference_scene_center, (3,)),
                  (msg.rate_scales, (2,)), (msg.linearization_rates, (2,)), (msg.a, (2,)),
                  (msg.joint_rate_matrix, (12,)), (msg.joint_rate_bias, (6,)),
                  (msg.joint_velocity_lower, (6,)), (msg.joint_velocity_upper, (6,)),
                  (msg.camera_linear_matrix, (6,)), (msg.camera_linear_bias, (3,)),
                  (msg.previous_linear, (3,)), (msg.joint_position, (6,)), (msg.joint_velocity, (6,)),
                  (msg.joint_position_lower, (6,)), (msg.joint_position_upper, (6,)),
                  (msg.joint_acceleration_lower, (6,)), (msg.joint_acceleration_upper, (6,)))
        if (not all(math.isfinite(float(value)) for value in scalars)
                or any(np.asarray(value).shape != shape or not np.all(np.isfinite(value))
                       for value, shape in arrays)):
            return "constraint_nonfinite_or_shape"
        if (msg.servo_dt_sec <= 0 or len(msg.joint_position_bounded) != 6
                or len(msg.joint_acceleration_bounded) != 6):
            return "constraint_joint_cycle_invalid"
        position_rows = np.asarray(msg.joint_position_bounded, dtype=bool)
        accel_rows = np.asarray(msg.joint_acceleration_bounded, dtype=bool)
        if (np.any(np.asarray(msg.joint_position_lower)[position_rows] >=
                   np.asarray(msg.joint_position_upper)[position_rows])
                or np.any(np.asarray(msg.joint_acceleration_lower)[accel_rows] >= 0)
                or np.any(np.asarray(msg.joint_acceleration_upper)[accel_rows] <= 0)):
            return "constraint_joint_dynamic_bounds_invalid"
        if (np.any(np.asarray(msg.joint_position)[position_rows] < np.asarray(msg.joint_position_lower)[position_rows])
                or np.any(np.asarray(msg.joint_position)[position_rows] > np.asarray(msg.joint_position_upper)[position_rows])):
            return "constraint_joint_position_outside_bounds"
        if min(msg.linear_speed_limit, msg.linear_step_limit, msg.linear_deadband) < 0:
            return "constraint_linear_limits_invalid"
        if (np.any(np.asarray(msg.joint_velocity_lower) >= 0)
                or np.any(np.asarray(msg.joint_velocity_upper) <= 0)):
            return "constraint_joint_bounds_invalid"
        if (min(radius, msg.reference_radius, msg.m_min, msg.alpha) <= 0 or msg.sigma_min < 0
                or min(msg.rate_scales) <= 0 or msg.derivative_error < 0
                or not 0 <= msg.inverse_residual <= 1e-6):
            return "constraint_metrics_invalid"
        expected_c = -msg.alpha*(msg.sigma_min-msg.m_min)-msg.drift
        if not math.isclose(msg.c, expected_c, rel_tol=1e-8, abs_tol=1e-10):
            return "constraint_bound_inconsistent"
        command = self.command
        key = (int(msg.planner_tick), int(msg.model_version), float(msg.command_stamp.to_sec()))
        if command is None or key != command["key"] or key[2] <= 0:
            return "constraint_execution_command_unmatched"
        if command["stop"] or command["frame"] != self.source_frame:
            return "constraint_execution_command_stopped_or_frame"
        if not np.allclose(msg.linearization_rates, command["rates"], rtol=1e-9, atol=1e-12):
            return "constraint_execution_rates_mismatch"
        if int(model_version) != int(msg.model_version):
            return "constraint_map_version_mismatch"
        if (not math.isclose(radius, msg.reference_radius, rel_tol=1e-9, abs_tol=1e-12)
                or not math.isclose(command["radius"], msg.reference_radius, rel_tol=1e-9, abs_tol=1e-12)
                or not np.allclose(center, msg.reference_scene_center, rtol=1e-9, atol=1e-12)
                or not np.allclose(command["center"], msg.reference_scene_center, rtol=1e-9, atol=1e-12)):
            return "constraint_reference_sphere_mismatch"
        return ""
