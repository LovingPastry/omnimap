"""ROS-independent validation of execution-side MoveIt Noetic feedback."""

from dataclasses import dataclass
import math


MOVEIT_STATUS_TEXT = {
    0: "NO_WARNING",
    1: "DECELERATE_FOR_SINGULARITY",
    2: "HALT_FOR_SINGULARITY",
    3: "DECELERATE_FOR_COLLISION",
    4: "HALT_FOR_COLLISION",
    5: "JOINT_BOUND",
}


@dataclass(frozen=True)
class FeedbackDecision:
    speed_scale: float = 0.0
    should_stop: bool = True
    reason: str = "kinematic_status_missing"


def validate_feedback(msg, *, received_wall, ros_now, wall_now, timeout_sec,
                      source_frame, task_frame):
    """Return an explicit failure reason, or an empty string for usable feedback.

    Both clocks matter: receipt time detects a stopped publisher even under a
    paused ROS clock, and source time detects delayed/replayed messages.
    """
    if msg is None:
        return "kinematic_status_missing"
    stamp = float(msg.header.stamp.to_sec())
    ages = (ros_now - stamp, wall_now - received_wall)
    if stamp <= 0.0 or not all(math.isfinite(v) for v in (stamp, *ages)):
        return "kinematic_status_invalid_stamp"
    if ages[0] < -0.05 or ages[1] < 0.0:
        return "kinematic_status_future"
    if max(ages) > timeout_sec:
        return "kinematic_status_stale"
    if msg.source_frame.lstrip("/") != source_frame.lstrip("/"):
        return "kinematic_source_frame_mismatch"
    if msg.task_frame.lstrip("/") != task_frame.lstrip("/"):
        return "kinematic_task_frame_mismatch"
    if not msg.transform_chain_valid:
        return "kinematic_transform_chain_invalid"
    metrics = (msg.sigma_min, msg.sigma_max, msg.manipulability,
               msg.command_speed_norm, msg.actual_speed_norm)
    if not all(math.isfinite(v) and v >= 0.0 for v in metrics):
        return "kinematic_metrics_invalid"
    if msg.sigma_max < msg.sigma_min:
        return "kinematic_metrics_invalid"
    condition = float(msg.condition_number)
    # A genuinely singular Jacobian may have an infinite condition number.
    if not ((math.isfinite(condition) and condition >= 1.0) or
            (condition == math.inf and msg.sigma_min == 0.0)):
        return "kinematic_metrics_invalid"
    return ""


class KinematicFeedbackGate:
    """Cache feedback and require a sustained deceleration before scaling.

    The caller serializes update/decision with its own lock. No Jacobian or
    correction direction is inferred from the status code.
    """

    def __init__(self, *, source_frame="base_link", task_frame="tool0",
                 timeout_sec=0.5, deceleration_duration_sec=0.1,
                 deceleration_scale=0.5):
        if not source_frame.lstrip("/") or not task_frame.lstrip("/"):
            raise ValueError("kinematic frames must be nonempty")
        if not math.isfinite(timeout_sec) or timeout_sec <= 0.0:
            raise ValueError("kinematic timeout must be positive and finite")
        if not math.isfinite(deceleration_duration_sec) or deceleration_duration_sec < 0.0:
            raise ValueError("deceleration duration must be nonnegative and finite")
        if not math.isfinite(deceleration_scale) or not 0.0 < deceleration_scale < 1.0:
            raise ValueError("deceleration scale must be between zero and one")
        self.source_frame = source_frame
        self.task_frame = task_frame
        self.timeout_sec = timeout_sec
        self.deceleration_duration_sec = deceleration_duration_sec
        self.deceleration_scale = deceleration_scale
        self.latest = None
        self.received_wall = 0.0
        self.decelerating_since = None

    def _validate(self, msg, received_wall, ros_now, wall_now):
        return validate_feedback(
            msg, received_wall=received_wall, ros_now=ros_now, wall_now=wall_now,
            timeout_sec=self.timeout_sec, source_frame=self.source_frame,
            task_frame=self.task_frame,
        )

    def update(self, msg, *, ros_now, wall_now):
        previous_valid = not self._validate(
            self.latest, self.received_wall, ros_now, wall_now,
        )
        valid = not self._validate(msg, wall_now, ros_now, wall_now)
        if not valid or msg.moveit_status != 1:
            self.decelerating_since = None
        elif not previous_valid or self.decelerating_since is None:
            self.decelerating_since = wall_now
        self.latest = msg
        self.received_wall = wall_now

    def decision(self, *, ros_now, wall_now):
        reason = self._validate(self.latest, self.received_wall, ros_now, wall_now)
        if reason:
            return FeedbackDecision(reason=reason)
        status = int(self.latest.moveit_status)
        if status not in (0, 1):
            name = MOVEIT_STATUS_TEXT.get(status, "UNKNOWN")
            return FeedbackDecision(reason=f"moveit_{name.lower()}:{status}")
        if (status == 1 and self.decelerating_since is not None and
                wall_now - self.decelerating_since >= self.deceleration_duration_sec):
            return FeedbackDecision(self.deceleration_scale, False,
                                    "moveit_decelerate_for_singularity")
        return FeedbackDecision(1.0, False, "kinematic_ok")
