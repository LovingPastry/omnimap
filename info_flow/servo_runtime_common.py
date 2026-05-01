from __future__ import annotations

import logging
import os
import resource
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import rospy
import tf2_ros
from geometry_msgs.msg import TransformStamped, TwistStamped
from scipy.spatial.transform import Rotation as R

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

REPO_ROOT = Path(__file__).resolve().parent.parent
ROS_WS_DEVEL = REPO_ROOT / "ros_ws" / "devel" / "lib" / "python3" / "dist-packages"
if ROS_WS_DEVEL.exists():
    ros_ws_path = str(ROS_WS_DEVEL)
    if ros_ws_path not in sys.path:
        sys.path.insert(0, ros_ws_path)

_LOGGING_CONFIGURED = False
_LOG_ROOT_NAME = "omnimap"
_SECTION_COLORS = {
    "main": "\033[38;5;117m",
    "planner": "\033[38;5;153m",
    "profile": "\033[38;5;222m",
}
_LEVEL_COLORS = {
    "DEBUG": "\033[38;5;151m",
    "INFO": "\033[38;5;111m",
    "WARNING": "\033[38;5;221m",
    "ERROR": "\033[38;5;203m",
}
_ANSI_RESET = "\033[0m"
_ANSI_BOLD = "\033[1m"


def set_nofile_limit() -> None:
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (100000, rlimit[1]))


class _ServoFormatter(logging.Formatter):
    def __init__(self, *args, enable_color: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.enable_color = bool(enable_color)

    def format(self, record: logging.LogRecord) -> str:
        if not hasattr(record, "section"):
            setattr(record, "section", "main")
        if not self.enable_color:
            return super().format(record)

        level_name = str(record.levelname).upper()
        section_name = str(getattr(record, "section", "main")).lower()
        level_color = _LEVEL_COLORS.get(level_name, "")
        section_color = _SECTION_COLORS.get(section_name, "")
        orig_levelname = record.levelname
        orig_section = record.section
        try:
            record.levelname = f"{_ANSI_BOLD}{level_color}{orig_levelname}{_ANSI_RESET}"
            record.section = f"{_ANSI_BOLD}{section_color}{orig_section}{_ANSI_RESET}"
            return super().format(record)
        finally:
            record.levelname = orig_levelname
            record.section = orig_section


class _SectionLoggerAdapter(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        extra = kwargs.setdefault("extra", {})
        extra.setdefault("section", self.extra["section"])
        return msg, kwargs


def configure_servo_logging(
    *,
    level: str = "INFO",
    log_file: Optional[str] = None,
    force: bool = False,
) -> logging.Logger:
    global _LOGGING_CONFIGURED
    logger = logging.getLogger(_LOG_ROOT_NAME)
    if _LOGGING_CONFIGURED and not force:
        return logger

    log_level = getattr(logging, str(level).upper(), logging.INFO)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    stream = logging.StreamHandler()
    stream.setLevel(log_level)
    use_color = bool(getattr(stream.stream, "isatty", lambda: False)())
    formatter = _ServoFormatter(
        "%(asctime)s | %(levelname)s | %(section)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        enable_color=use_color,
    )
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    if log_file:
        log_dir = os.path.dirname(os.fspath(log_file))
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(os.fspath(log_file), encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)s | %(section)s | %(name)s | %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        logger.addHandler(file_handler)

    _LOGGING_CONFIGURED = True
    return logger


def get_section_logger(name: str, section: str):
    base_logger = logging.getLogger(f"{_LOG_ROOT_NAME}.{name}")
    return _SectionLoggerAdapter(base_logger, {"section": section})


def import_spherical_command_msg():
    try:
        from omnimap_msgs.msg import SphericalCommand
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "failed to import omnimap_msgs.SphericalCommand; build ros_ws and source ros_ws/devel/setup.bash first"
        ) from exc
    return SphericalCommand


def transform_to_c2w(transform: TransformStamped) -> np.ndarray:
    t = transform.transform.translation
    q = transform.transform.rotation
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, 3] = [t.x, t.y, t.z]
    c2w[:3, :3] = R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
    return c2w


def lookup_latest_pose_from_tf(
    tf_buffer: tf2_ros.Buffer,
    *,
    world_frame: str,
    camera_frame: str,
    timeout_sec: float = 0.05,
) -> Tuple[np.ndarray, rospy.Time]:
    transform: TransformStamped = tf_buffer.lookup_transform(
        world_frame,
        camera_frame,
        rospy.Time(0),
        rospy.Duration(timeout_sec),
    )
    return transform_to_c2w(transform), transform.header.stamp


def publish_zero_twist(pub, *, cmd_frame: str, stamp: Optional[rospy.Time] = None) -> None:
    msg = TwistStamped()
    msg.header.stamp = stamp if stamp is not None else rospy.Time.now()
    msg.header.frame_id = cmd_frame
    pub.publish(msg)


def publish_motion_components(
    pub,
    *,
    cmd_frame: str,
    linear_cmd: np.ndarray,
    angular_cmd: np.ndarray,
    stamp: Optional[rospy.Time] = None,
) -> None:
    msg = TwistStamped()
    msg.header.stamp = stamp if stamp is not None else rospy.Time.now()
    msg.header.frame_id = cmd_frame
    linear = np.asarray(linear_cmd, dtype=np.float64).reshape(3)
    angular = np.asarray(angular_cmd, dtype=np.float64).reshape(3)
    msg.twist.linear.x = float(linear[0])
    msg.twist.linear.y = float(linear[1])
    msg.twist.linear.z = float(linear[2])
    msg.twist.angular.x = float(angular[0])
    msg.twist.angular.y = float(angular[1])
    msg.twist.angular.z = float(angular[2])
    pub.publish(msg)


def position_to_spherical(
    position: np.ndarray,
    scene_center: np.ndarray,
) -> Tuple[float, float, float]:
    offset = np.asarray(position, dtype=np.float64).reshape(3) - np.asarray(
        scene_center,
        dtype=np.float64,
    ).reshape(3)
    radius = float(np.linalg.norm(offset))
    if radius < 1e-12:
        raise ValueError("position is too close to scene center")
    n_hat = offset / radius
    theta = float(np.arctan2(n_hat[1], n_hat[0]) % (2.0 * np.pi))
    phi = float(np.arcsin(np.clip(n_hat[2], 0.0, 1.0)))
    return radius, theta, phi


def local_frame_from_theta_phi(theta: float, phi: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ct, st = float(np.cos(theta)), float(np.sin(theta))
    cp, sp = float(np.cos(phi)), float(np.sin(phi))
    e_theta = np.array([-cp * st, cp * ct, 0.0], dtype=np.float64)
    e_phi = np.array([-sp * ct, -sp * st, cp], dtype=np.float64)
    n_hat = np.array([cp * ct, cp * st, sp], dtype=np.float64)
    return e_theta, e_phi, n_hat


def look_at_c2w(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
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
