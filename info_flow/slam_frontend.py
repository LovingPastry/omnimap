from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from threading import Condition
from typing import Deque, Generic, Optional, TypeVar

import numpy as np
import time

T = TypeVar("T")


@dataclass
class KeyframeDecision:
    should_track: bool
    reason: str
    dt_sec: float
    translation_m: float
    rotation_deg: float


class RTabMapKeyframeGate:
    """Autonomous keyframe filter: time/motion thresholds + forced-gap pass."""

    def __init__(
        self,
        *,
        min_interval_sec: float,
        min_translation_m: float,
        min_rotation_deg: float,
        forced_gap_frames: int = 30,
    ) -> None:
        self.min_interval_sec = float(min_interval_sec)
        self.min_translation_m = float(min_translation_m)
        self.min_rotation_deg = float(min_rotation_deg)
        self.forced_gap_frames = max(1, int(forced_gap_frames))
        self._last_accept_stamp_sec: Optional[float] = None
        self._last_accept_pose_4x4: Optional[np.ndarray] = None
        self._consecutive_rejects = 0

    @staticmethod
    def _rotation_deg_between(prev_pose: np.ndarray, curr_pose: np.ndarray) -> float:
        prev_r = np.asarray(prev_pose, dtype=np.float64)[:3, :3]
        curr_r = np.asarray(curr_pose, dtype=np.float64)[:3, :3]
        rel = prev_r.T @ curr_r
        trace_val = float(np.trace(rel))
        cos_theta = max(-1.0, min(1.0, (trace_val - 1.0) * 0.5))
        return float(np.degrees(math.acos(cos_theta)))

    def decide(
        self, *, pose_4x4: np.ndarray, stamp_sec: float, frame_index: int
    ) -> KeyframeDecision:
        stamp_sec = float(stamp_sec)
        pose_4x4 = np.asarray(pose_4x4, dtype=np.float64)

        if self._last_accept_stamp_sec is None or self._last_accept_pose_4x4 is None:
            self._last_accept_stamp_sec = stamp_sec
            self._last_accept_pose_4x4 = pose_4x4
            self._consecutive_rejects = 0
            return KeyframeDecision(
                should_track=True,
                reason="first_frame",
                dt_sec=0.0,
                translation_m=0.0,
                rotation_deg=0.0,
            )

        dt_sec = max(0.0, stamp_sec - float(self._last_accept_stamp_sec))
        prev_pose = np.asarray(self._last_accept_pose_4x4, dtype=np.float64)
        translation_m = float(
            np.linalg.norm(pose_4x4[:3, 3] - np.asarray(prev_pose[:3, 3], dtype=np.float64))
        )
        rotation_deg = self._rotation_deg_between(prev_pose, pose_4x4)

        reject_reason = None
        if dt_sec < self.min_interval_sec:
            reject_reason = "below_min_interval"
        elif (
            translation_m < self.min_translation_m
            and rotation_deg < self.min_rotation_deg
        ):
            reject_reason = "below_motion_threshold"

        if reject_reason is not None:
            self._consecutive_rejects += 1
            if self._consecutive_rejects >= self.forced_gap_frames:
                self._last_accept_stamp_sec = stamp_sec
                self._last_accept_pose_4x4 = pose_4x4
                self._consecutive_rejects = 0
                return KeyframeDecision(
                    should_track=True,
                    reason="forced_gap_keyframe",
                    dt_sec=dt_sec,
                    translation_m=translation_m,
                    rotation_deg=rotation_deg,
                )
            return KeyframeDecision(
                should_track=False,
                reason=reject_reason,
                dt_sec=dt_sec,
                translation_m=translation_m,
                rotation_deg=rotation_deg,
            )

        self._last_accept_stamp_sec = stamp_sec
        self._last_accept_pose_4x4 = pose_4x4
        self._consecutive_rejects = 0
        return KeyframeDecision(
            should_track=True,
            reason="accepted",
            dt_sec=dt_sec,
            translation_m=translation_m,
            rotation_deg=rotation_deg,
        )


class DropOldestQueue(Generic[T]):
    """Thread-safe bounded queue that drops the oldest item when full."""

    def __init__(self, maxsize: int) -> None:
        if int(maxsize) <= 0:
            raise ValueError(f"maxsize must be positive, got {maxsize}")
        self.maxsize = int(maxsize)
        self._items: Deque[T] = deque()
        self._cond = Condition()
        self._closed = False

    def put(self, item: T) -> Optional[T]:
        """Push one item and optionally return the dropped oldest item."""
        with self._cond:
            dropped = None
            if len(self._items) >= self.maxsize:
                dropped = self._items.popleft()
            self._items.append(item)
            self._cond.notify()
            return dropped

    def get(self, timeout: Optional[float] = None) -> T:
        with self._cond:
            if timeout is None:
                while not self._items and not self._closed:
                    self._cond.wait()
            else:
                deadline = timeout + time.monotonic()
                while not self._items and not self._closed:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("queue get timeout")
                    self._cond.wait(remaining)

            if self._items:
                return self._items.popleft()
            raise TimeoutError("queue closed")

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._cond.notify_all()

    def qsize(self) -> int:
        with self._cond:
            return len(self._items)

    def empty(self) -> bool:
        return self.qsize() == 0
