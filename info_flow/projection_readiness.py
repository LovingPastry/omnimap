"""Thread-safe health/recovery gate for local projection results."""
import copy
import math
import threading
from dataclasses import dataclass
from kinematic_feedback import validate_feedback


@dataclass(frozen=True)
class ProjectionReadiness:
    ready: bool
    state: str
    reason: str


class ProjectionReadinessGate:
    def __init__(self, *, source_frame="base_link", task_frame="tool0",
                 timeout_sec=.2, recovery_sec=.1):
        if (not math.isfinite(timeout_sec) or timeout_sec <= 0
                or not math.isfinite(recovery_sec) or recovery_sec < 0):
            raise ValueError("invalid projection readiness timing")
        self.source_frame, self.task_frame = source_frame, task_frame
        self.timeout_sec, self.recovery_sec = timeout_sec, recovery_sec
        self._lock = threading.Lock()
        self._latest, self._received = None, 0.
        self._since, self._count, self._last_stamp = None, 0, 0.
        self._stream_error = ""

    def _error(self, msg, received, ros_now, wall_now):
        error = validate_feedback(msg, received_wall=received, ros_now=ros_now, wall_now=wall_now,
                                  timeout_sec=self.timeout_sec, source_frame=self.source_frame,
                                  task_frame=self.task_frame)
        if not error and msg.moveit_status not in (0, 1):
            error = "projection_moveit_stop:%s" % msg.moveit_status
        return error

    def update(self, msg, *, ros_now, wall_now):
        msg = copy.deepcopy(msg)
        with self._lock:
            if self._error(self._latest, self._received, ros_now, wall_now) or self._stream_error:
                self._since, self._count = None, 0
            error = self._error(msg, wall_now, ros_now, wall_now)
            stamp = float(msg.header.stamp.to_sec()) if msg is not None else 0.
            # A newer stop/invalid-chain sample is still a temporal barrier:
            # older healthy feedback must not restart recovery across it.
            stamp_usable = (math.isfinite(stamp) and stamp > 0 and math.isfinite(ros_now)
                            and -.05 <= ros_now-stamp <= self.timeout_sec)
            if stamp_usable:
                if stamp <= self._last_stamp:
                    error = "projection_status_stamp_did_not_advance"
                else:
                    self._last_stamp = stamp
            self._latest, self._received, self._stream_error = msg, wall_now, error
            if error:
                self._since, self._count = None, 0
            else:
                if self._since is None:
                    self._since = wall_now
                self._count += 1

    def decision(self, *, ros_now, wall_now):
        with self._lock:
            error = self._error(self._latest, self._received, ros_now, wall_now) or self._stream_error
            if error:
                self._since, self._count = None, 0
                return ProjectionReadiness(False, "blocked", error)
            if self._since is None or self._count < 2 or wall_now-self._since < self.recovery_sec:
                return ProjectionReadiness(False, "recovering", "projection_status_recovering")
            return ProjectionReadiness(True, "ready", "projection_status_ready")
