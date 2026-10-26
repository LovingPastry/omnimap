"""Local JointState corroboration for a frozen, non-actuating model preview."""

import math
import threading
import numpy as np


class PreviewJointState:
    def __init__(self):
        self._lock = threading.Lock()
        self._latest = None
        self._last_stamp = 0.
        self._error = "preview local joint state missing"

    def update(self, msg, wall_now, *, ros_now=None):
        with self._lock:
            self._latest = None
            try:
                stamp = msg.header.stamp.to_sec()
                names = tuple(msg.name)
                q, dq = np.asarray(msg.position, dtype=float), np.asarray(msg.velocity, dtype=float)
                if (not names or len(set(names)) != len(names) or any(not n.strip() for n in names)
                        or q.shape != (len(names),) or dq.shape != q.shape
                        or not np.all(np.isfinite(q)) or not np.all(np.isfinite(dq))
                        or not math.isfinite(stamp) or stamp <= 0 or not math.isfinite(wall_now)):
                    raise ValueError("preview local joint state malformed")
                if ros_now is not None and (not math.isfinite(ros_now) or stamp-ros_now > .05):
                    raise ValueError("preview local joint stamp is in the future")
                if stamp <= self._last_stamp:
                    raise ValueError("preview local joint stamp did not advance")
                self._last_stamp = stamp
                self._latest = (stamp, float(wall_now), dict(zip(names, zip(q.tolist(), dq.tolist()))))
                self._error = ""
            except (ValueError, TypeError) as exc:
                self._error = str(exc)

    def validate(self, response, *, ros_now, wall_now, timeout):
        with self._lock:
            sample, error = self._latest, self._error
        if sample is None:
            raise ValueError(error)
        stamp, received, joints = sample
        if (not all(math.isfinite(x) for x in (ros_now, wall_now, timeout)) or timeout <= 0
                or not -.05 <= ros_now-stamp <= timeout or not 0 <= wall_now-received <= timeout):
            raise ValueError("preview local joint state stale")
        names = tuple(response.joint_names)
        q, dq = np.asarray(response.joint_position), np.asarray(response.joint_velocity)
        if (len(names) != 6 or len(set(names)) != 6 or any(name not in joints for name in names)
                or q.shape != (6,) or dq.shape != (6,)
                or not np.all(np.isfinite(q)) or not np.all(np.isfinite(dq))):
            raise ValueError("preview response joint sample invalid")
        source_stamp = response.joint_stamp.to_sec()
        if not math.isfinite(source_stamp) or source_stamp <= 0 or stamp < source_stamp:
            raise ValueError("preview local joint sample precedes response")
        actual = np.asarray([joints[name] for name in names])
        if (not np.allclose(actual[:, 0], q, rtol=0, atol=1e-9)
                or not np.allclose(actual[:, 1], dq, rtol=0, atol=1e-9)):
            raise ValueError("preview joint state changed since model sample")
