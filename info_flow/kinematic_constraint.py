"""Validate execution-side local models and compute non-actuating proposals.

This consumer deliberately has no enforce mode: the producer has not yet
provided the joint/speed/acceleration feasible set or a stop/recovery model.
"""
from dataclasses import dataclass
import copy
import math

import numpy as np

from kinematic_projection import project_spherical_command


@dataclass(frozen=True)
class ProjectionProposal:
    valid: bool = False
    reason: str = "constraint_missing"
    rates: tuple = (0.0, 0.0)
    correction_norm: float = 0.0
    source_tick: int = 0


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
        # Store only final execution commands, not the raw Fisher diagnostic.
        # A later stop/replan invalidates an earlier command's model immediately.
        self.command = dict(
            key=(int(msg.planner_tick), int(msg.model_version), float(msg.header.stamp.to_sec())),
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
        if np.linalg.norm(original-origin) > self.max_normalized_delta:
            return ProjectionProposal(reason="constraint_candidate_outside_local_radius")
        try:
            projected, norm = project_spherical_command(original, msg.a, msg.c)
        except ValueError as exc:
            return ProjectionProposal(reason="constraint_projection_unavailable:" + str(exc))
        if np.linalg.norm(projected-origin) > self.max_normalized_delta:
            return ProjectionProposal(reason="constraint_projection_outside_local_radius")
        return ProjectionProposal(True, "advisory_projection", tuple(projected * scales),
                                  norm, int(msg.planner_tick))

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
        if not msg.threshold_configured:
            return "constraint_threshold_unset"
        scalars = (radius, msg.reference_radius, msg.sigma_min, msg.m_min, msg.alpha,
                   msg.c, msg.drift, msg.inverse_residual, msg.derivative_error)
        arrays = ((rates, (2,)), (center, (3,)), (msg.reference_scene_center, (3,)),
                  (msg.rate_scales, (2,)), (msg.linearization_rates, (2,)), (msg.a, (2,)))
        if (not all(math.isfinite(float(value)) for value in scalars)
                or any(np.asarray(value).shape != shape or not np.all(np.isfinite(value))
                       for value, shape in arrays)):
            return "constraint_nonfinite_or_shape"
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
