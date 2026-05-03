"""Fisher / NBV motion policy shared by simulation and real-world wrappers."""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from typing import Any, Optional, Sequence

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from omnimap.util.utils import get_section_logger


def _as_matrix44(matrix: np.ndarray, name: str) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"{name} must have shape (4, 4), got {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} contains NaN or Inf")
    return matrix


def c2w_to_w2c(c2w: np.ndarray) -> np.ndarray:
    return np.linalg.inv(_as_matrix44(c2w, "c2w"))


def w2c_to_c2w(w2c: np.ndarray) -> np.ndarray:
    return np.linalg.inv(_as_matrix44(w2c, "w2c"))


def _look_at_c2w(
    eye: np.ndarray,
    target: np.ndarray,
    up: np.ndarray | None = None,
) -> np.ndarray:
    # 统一输入形状/类型：eye 是相机位置，target 是注视点，up 是世界“上”方向参考。
    eye = np.asarray(eye, dtype=np.float64).reshape(3)
    target = np.asarray(target, dtype=np.float64).reshape(3)
    up = (
        np.array([0.0, 0.0, 1.0], dtype=np.float64)
        if up is None
        else np.asarray(up, dtype=np.float64).reshape(3)
    )

    # 相机前向轴（z 轴）指向 target。
    forward = target - eye
    forward_norm = np.linalg.norm(forward)
    if forward_norm < 1e-12:
        raise ValueError("eye and target are too close; cannot build look-at pose")
    forward = forward / forward_norm

    # 右轴 = forward × up；若与 up 近共线，退化到备用 up 再算一次。
    right = np.cross(forward, up)
    right_norm = np.linalg.norm(right)
    if right_norm < 1e-12:
        fallback_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        right = np.cross(forward, fallback_up)
        right_norm = np.linalg.norm(right)
        if right_norm < 1e-12:
            raise ValueError("failed to construct a valid right axis for look-at pose")
    right = right / right_norm

    # 由 right 与 forward 反推真实 up，保持三个轴两两正交。
    true_up = np.cross(right, forward)
    true_up = true_up / max(np.linalg.norm(true_up), 1e-12)
    # 本项目相机坐标约定 y 轴朝下，因此第二列使用 -up。
    down = -true_up

    # 组装 c2w：前三列分别是相机 x/y/z 轴在世界系中的方向，最后一列是相机位置。
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, 0] = right
    c2w[:3, 1] = down
    c2w[:3, 2] = forward
    c2w[:3, 3] = eye
    return c2w


def _look_at_c2w_min_roll_current(
    eye: np.ndarray,
    target: np.ndarray,
    current_c2w: np.ndarray,
) -> np.ndarray:
    """Build a look-at pose while preserving the current roll as much as possible."""
    eye = np.asarray(eye, dtype=np.float64).reshape(3)
    target = np.asarray(target, dtype=np.float64).reshape(3)
    current_c2w = _as_matrix44(current_c2w, "current_c2w")

    forward = target - eye
    forward_norm = np.linalg.norm(forward)
    if forward_norm < 1e-12:
        raise ValueError("eye and target are too close; cannot build look-at pose")
    forward = forward / forward_norm

    def _project_to_forward_plane(axis: np.ndarray) -> np.ndarray:
        axis = np.asarray(axis, dtype=np.float64).reshape(3)
        return axis - float(np.dot(axis, forward)) * forward

    # Prefer current x-axis (right) to preserve roll around the forward axis.
    candidate_axes = (
        current_c2w[:3, 0],  # current right
        -current_c2w[:3, 1],  # current up (because y column stores down)
        np.array([1.0, 0.0, 0.0], dtype=np.float64),
        np.array([0.0, 1.0, 0.0], dtype=np.float64),
    )
    right = None
    for axis in candidate_axes:
        proj = _project_to_forward_plane(axis)
        proj_norm = float(np.linalg.norm(proj))
        if proj_norm > 1e-12:
            right = proj / proj_norm
            break
    if right is None:
        raise ValueError("failed to construct a valid right axis for look-at pose")

    true_up = np.cross(right, forward)
    up_norm = float(np.linalg.norm(true_up))
    if up_norm < 1e-12:
        raise ValueError("failed to construct a valid up axis for look-at pose")
    true_up = true_up / up_norm
    down = -true_up

    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, 0] = right
    c2w[:3, 1] = down
    c2w[:3, 2] = forward
    c2w[:3, 3] = eye
    return c2w


def _spherical_c2w(
    *,
    scene_center: np.ndarray,
    radius: float,
    theta: float,
    phi: float,
) -> np.ndarray:
    center = np.asarray(scene_center, dtype=np.float64).reshape(3)
    eye = center + np.array(
        [
            radius * math.cos(phi) * math.cos(theta),
            radius * math.cos(phi) * math.sin(theta),
            radius * math.sin(phi),
        ],
        dtype=np.float64,
    )
    return _look_at_c2w(eye=eye, target=center)


def _spherical_direction(theta: float, phi: float) -> np.ndarray:
    return np.array(
        [
            math.cos(phi) * math.cos(theta),
            math.cos(phi) * math.sin(theta),
            math.sin(phi),
        ],
        dtype=np.float64,
    )


def _local_frame_from_theta_phi(
    theta: float,
    phi: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ct = math.cos(theta)
    st = math.sin(theta)
    cp = math.cos(phi)
    sp = math.sin(phi)
    e_theta = np.array([-cp * st, cp * ct, 0.0], dtype=np.float64)
    e_phi = np.array([-sp * ct, -sp * st, cp], dtype=np.float64)
    n_hat = np.array([cp * ct, cp * st, sp], dtype=np.float64)
    return e_theta, e_phi, n_hat


def _position_to_spherical(
    position: np.ndarray,
    scene_center: np.ndarray,
) -> tuple[float, float, float]:
    offset = np.asarray(position, dtype=np.float64).reshape(3) - np.asarray(
        scene_center, dtype=np.float64
    ).reshape(3)
    radius = float(np.linalg.norm(offset))
    if radius < 1e-12:
        raise ValueError(
            "position is too close to scene_center; cannot recover spherical state"
        )
    n_hat = offset / radius
    theta = float(math.atan2(n_hat[1], n_hat[0]) % (2.0 * math.pi))
    phi = float(math.asin(np.clip(n_hat[2], 0.0, 1.0)))
    return radius, theta, phi


@dataclass
class MotionPolicyResult:
    idx: int
    viewpoint_source: str
    controller_mode: str
    cartesian: bool
    current_c2w: np.ndarray
    next_c2w: np.ndarray
    scene_center: list[float]
    reference_scene_center: list[float]
    look_at_target: list[float]
    desired_c2w: np.ndarray
    radius: float
    reference_radius: float
    current_radius: float
    radial_error: float
    dt: float
    current_theta: float
    current_phi: float
    next_theta: float
    next_phi: float
    grad_theta_raw: float
    grad_phi_raw: float
    grad_norm_raw: float
    grad_theta_compressed: float
    grad_phi_compressed: float
    grad_norm_compressed: float
    num_gaussians: int
    fisher_score: float
    scaled_theta: float
    scaled_phi: float
    delta_theta_applied: float
    delta_phi_applied: float
    velocity_raw_world: np.ndarray
    vt_world: np.ndarray
    vn_world: np.ndarray
    velocity_world: np.ndarray
    rotvec_error: np.ndarray
    angular_velocity_world: np.ndarray
    angular_speed_raw: float
    angular_speed_applied: float
    angular_gain: float
    enable_angular: bool
    next_position: list[float]
    step_scale_theta: float
    step_scale_phi: float
    linear_speed_limit: float
    linear_speed_raw: float
    linear_speed_applied: float
    spherical_speed_limit: float
    spherical_speed_min: float
    spherical_speed_raw: float
    spherical_speed_scaled: float
    spherical_speed_applied: float
    clip_scale_ratio: float
    speed_clipped: bool
    should_stop: bool
    stop_reason: str
    planner_output_mode: str = "cartesian_legacy"

    def to_jsonable(self) -> dict[str, object]:
        data = asdict(self)
        for key, value in list(data.items()):
            if isinstance(value, np.ndarray):
                data[key] = value.tolist()
        return data

    @property
    def grad_theta(self) -> float:
        return self.grad_theta_raw

    @property
    def grad_phi(self) -> float:
        return self.grad_phi_raw

    @property
    def grad_norm(self) -> float:
        return self.grad_norm_raw


class FisherMotionPolicy:
    """Shared Fisher-driven motion policy for simulation and real-world wrappers."""

    _SUPPORTED_ORIENTATION_ROLL_MODES = {
        "world_up_lookat",
        "current_frame_min_roll",
    }
    _SUPPORTED_CONTROL_LAW_MODES = {
        "gain",
        "dt_consistent",
    }
    _SUPPORTED_PLANNER_OUTPUT_MODES = {
        "cartesian_legacy",
        "spherical_delta",
    }

    def __init__(
        self,
        fisher_step_scale: float = 0.03,
        *,
        cartesian: bool = False,
        dt: float = 0.1,
        radial_gain: float = 2.0,
        linear_vel_max: float = 0.5,
        angular_gain: float = 2.0,
        angular_speed_max: float = 1.0,
        enable_angular: bool = True,
        grad_eps: float = 0.01,
        spherical_speed_min: float = 1e-4,
        max_delta_theta: float = 0.20,
        max_delta_phi: float = 0.15,
        phi_min: float = 1e-3,
        phi_max: float | None = None,
        orientation_roll_mode: str = "world_up_lookat",
        control_law_mode: str = "gain",
        planner_output_mode: str = "cartesian_legacy",
        verbose: bool = True,
    ) -> None:
        if not np.isfinite(fisher_step_scale):
            raise ValueError("fisher_step_scale must be finite")
        if dt <= 0 or not np.isfinite(dt):
            raise ValueError(f"dt must be positive and finite, got {dt}")
        if radial_gain < 0 or not np.isfinite(radial_gain):
            raise ValueError(
                f"radial_gain must be non-negative and finite, got {radial_gain}"
            )
        if linear_vel_max <= 0 or not np.isfinite(linear_vel_max):
            raise ValueError(
                f"linear_vel_max must be positive and finite, got {linear_vel_max}"
            )
        if angular_gain < 0 or not np.isfinite(angular_gain):
            raise ValueError(
                f"angular_gain must be non-negative and finite, got {angular_gain}"
            )
        if angular_speed_max <= 0 or not np.isfinite(angular_speed_max):
            raise ValueError(
                f"angular_speed_max must be positive and finite, got {angular_speed_max}"
            )
        if grad_eps <= 0:
            raise ValueError(f"grad_eps must be positive, got {grad_eps}")
        if spherical_speed_min < 0 or not np.isfinite(spherical_speed_min):
            raise ValueError(
                f"spherical_speed_min must be non-negative and finite, got {spherical_speed_min}"
            )
        if max_delta_theta <= 0 or max_delta_phi <= 0:
            raise ValueError("max deltas must be positive")
        if orientation_roll_mode not in self._SUPPORTED_ORIENTATION_ROLL_MODES:
            raise ValueError(
                "orientation_roll_mode must be one of "
                f"{sorted(self._SUPPORTED_ORIENTATION_ROLL_MODES)}, "
                f"got {orientation_roll_mode!r}"
            )
        if control_law_mode not in self._SUPPORTED_CONTROL_LAW_MODES:
            raise ValueError(
                "control_law_mode must be one of "
                f"{sorted(self._SUPPORTED_CONTROL_LAW_MODES)}, "
                f"got {control_law_mode!r}"
            )
        if planner_output_mode not in self._SUPPORTED_PLANNER_OUTPUT_MODES:
            raise ValueError(
                "planner_output_mode must be one of "
                f"{sorted(self._SUPPORTED_PLANNER_OUTPUT_MODES)}, "
                f"got {planner_output_mode!r}"
            )

        self.fisher_step_scale = float(fisher_step_scale)
        self.controller_mode = "cartesian" if cartesian else "angular"
        self.cartesian = bool(cartesian)
        self.dt = float(dt)
        self.radial_gain = float(radial_gain)
        self.radial_deadband = 1e-3
        self.linear_vel_max = float(linear_vel_max)
        self.angular_gain = float(angular_gain)
        self.angular_speed_deadband = 1e-3
        self.angular_speed_max = float(angular_speed_max)
        self.enable_angular = bool(enable_angular)
        self.grad_eps = float(grad_eps)
        self.spherical_speed_min = float(spherical_speed_min)
        self.max_delta_theta = float(max_delta_theta)
        self.max_delta_phi = float(max_delta_phi)
        self.phi_min = float(phi_min)
        self.phi_max = float(math.pi / 2.0 - 1e-3 if phi_max is None else phi_max)
        self.orientation_roll_mode = str(orientation_roll_mode)
        self.control_law_mode = str(control_law_mode)
        self.planner_output_mode = str(planner_output_mode)
        self.verbose = bool(verbose)
        self.logger = get_section_logger("planner.motion_policy", "planner")
        self._omega_cmd_preclip_norm = 0.0
        self._vn_cmd_preclip_norm = 0.0
        self.reference_scene_center: Optional[np.ndarray] = None
        self.reference_radius: Optional[float] = None
        self.reference_initialized = False
        self.last_timing: dict[str, float] = {
            "history_ms": 0.0,
            "score_ms": 0.0,
            "gradient_ms": 0.0,
            "fisher_ms": 0.0,
            "s2c_ms": 0.0,
            "policy_total_ms": 0.0,
            "history_source": "missing",
        }

        if not (0.0 <= self.phi_min <= self.phi_max <= math.pi / 2.0):
            raise ValueError(
                f"invalid phi range: phi_min={self.phi_min}, phi_max={self.phi_max}"
            )

    @staticmethod
    def _wrap_theta(theta: float) -> float:
        return float(theta % (2.0 * math.pi))

    def _clamp_phi(self, phi: float) -> float:
        return float(np.clip(phi, self.phi_min, self.phi_max))

    @staticmethod
    def _clip_vector_norm(
        vec: np.ndarray,
        limit: float,
    ) -> tuple[np.ndarray, bool, float]:
        vec = np.asarray(vec, dtype=np.float64).reshape(2)
        norm = float(np.linalg.norm(vec))
        if norm <= limit or norm <= 1e-12:
            return vec, False, 1.0
        scale = float(limit / norm)
        return vec * scale, True, scale

    @staticmethod
    def _integrate_world_angular_velocity(
        current_rotation: np.ndarray,
        angular_velocity_world: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        current_rotation = np.asarray(current_rotation, dtype=np.float64).reshape(3, 3)
        angular_velocity_world = np.asarray(
            angular_velocity_world, dtype=np.float64
        ).reshape(3)
        delta_rotation = R.from_rotvec(angular_velocity_world * float(dt)).as_matrix()
        next_rotation = delta_rotation @ current_rotation
        u, _, vh = np.linalg.svd(next_rotation)
        return u @ vh

    def _compute_desired_orientation(
        self,
        *,
        current_position: np.ndarray,
        reference_scene_center: np.ndarray,
        current_c2w: np.ndarray | None = None,
    ) -> np.ndarray:
        # 期望姿态的定义：保持当前位置不变，让相机始终朝向参考场景中心。
        eye = np.asarray(current_position, dtype=np.float64)
        target = np.asarray(reference_scene_center, dtype=np.float64)
        if self.orientation_roll_mode == "world_up_lookat":
            return _look_at_c2w(
                eye=eye,
                target=target,
            )
        if current_c2w is None:
            raise ValueError(
                "current_c2w is required when orientation_roll_mode="
                "'current_frame_min_roll'"
            )
        return _look_at_c2w_min_roll_current(
            eye=eye,
            target=target,
            current_c2w=current_c2w,
        )

    @staticmethod
    def _build_stop_result(
        *,
        idx: int,
        viewpoint_source: str,
        controller_mode: str,
        cartesian: bool,
        current_c2w: np.ndarray,
        scene_center_np: np.ndarray,
        reference_scene_center: np.ndarray,
        look_at_target: np.ndarray,
        radius: float,
        reference_radius: float,
        current_radius: float,
        radial_error: float,
        dt: float,
        current_theta: float,
        current_phi: float,
        grad_theta: float,
        grad_phi: float,
        grad_norm: float,
        grad_theta_compressed: float,
        grad_phi_compressed: float,
        grad_norm_compressed: float,
        num_gaussians: int,
        fisher_score: float,
        scaled_spherical_velocity: np.ndarray,
        step_scale_theta: float,
        step_scale_phi: float,
        spherical_speed_min: float,
        spherical_speed_limit: float,
        speed_clipped: bool,
        clip_scale_ratio: float,
    ) -> MotionPolicyResult:
        return MotionPolicyResult(
            idx=int(idx),
            viewpoint_source=viewpoint_source,
            controller_mode=controller_mode,
            cartesian=cartesian,
            current_c2w=current_c2w,
            next_c2w=current_c2w.copy(),
            scene_center=np.asarray(scene_center_np, dtype=np.float64).tolist(),
            reference_scene_center=np.asarray(
                reference_scene_center, dtype=np.float64
            ).tolist(),
            look_at_target=np.asarray(look_at_target, dtype=np.float64).tolist(),
            desired_c2w=current_c2w.copy(),
            radius=float(radius),
            reference_radius=float(reference_radius),
            current_radius=float(current_radius),
            radial_error=float(radial_error),
            dt=float(dt),
            current_theta=float(current_theta),
            current_phi=float(current_phi),
            next_theta=float(current_theta),
            next_phi=float(current_phi),
            grad_theta_raw=float(grad_theta),
            grad_phi_raw=float(grad_phi),
            grad_norm_raw=float(grad_norm),
            grad_theta_compressed=float(grad_theta_compressed),
            grad_phi_compressed=float(grad_phi_compressed),
            grad_norm_compressed=float(grad_norm_compressed),
            num_gaussians=int(num_gaussians),
            fisher_score=float(fisher_score),
            scaled_theta=float(scaled_spherical_velocity[0]),
            scaled_phi=float(scaled_spherical_velocity[1]),
            delta_theta_applied=0.0,
            delta_phi_applied=0.0,
            velocity_raw_world=np.zeros(3, dtype=np.float64),
            vt_world=np.zeros(3, dtype=np.float64),
            vn_world=np.zeros(3, dtype=np.float64),
            velocity_world=np.zeros(3, dtype=np.float64),
            rotvec_error=np.zeros(3, dtype=np.float64),
            angular_velocity_world=np.zeros(3, dtype=np.float64),
            angular_speed_raw=0.0,
            angular_speed_applied=0.0,
            angular_gain=0.0,
            enable_angular=False,
            next_position=current_c2w[:3, 3].astype(np.float64).tolist(),
            step_scale_theta=float(step_scale_theta),
            step_scale_phi=float(step_scale_phi),
            linear_speed_limit=0.0,
            linear_speed_raw=0.0,
            linear_speed_applied=0.0,
            spherical_speed_limit=(
                float("nan") if cartesian else float(spherical_speed_limit)
            ),
            spherical_speed_min=float(spherical_speed_min),
            spherical_speed_raw=float(math.hypot(grad_theta, grad_phi)),
            spherical_speed_scaled=float(np.linalg.norm(scaled_spherical_velocity)),
            spherical_speed_applied=0.0,
            clip_scale_ratio=float(clip_scale_ratio),
            speed_clipped=bool(speed_clipped),
            should_stop=True,
            stop_reason="below_min_speed",
            planner_output_mode="cartesian_legacy",
        )

    @staticmethod
    def _extract_current_c2w(
        *,
        current_viewpoint: Any,
        viewpoint_source: str,
        current_theta: float,
        current_phi: float,
        scene_center_np: np.ndarray,
        radius: float,
    ) -> np.ndarray:
        if viewpoint_source == "c2w_matrix":
            c2w = np.asarray(current_viewpoint, dtype=np.float64)
            if c2w.shape != (4, 4):
                raise ValueError(f"expected 4x4 c2w matrix, got {c2w.shape}")
            return c2w
        if viewpoint_source in {"camera", "hemisphere_camera"}:
            w2c = FisherMotionPolicy._build_w2c44_from_camera(current_viewpoint)
            return w2c_to_c2w(w2c)
        return _spherical_c2w(
            scene_center=scene_center_np,
            radius=radius,
            theta=current_theta,
            phi=current_phi,
        )

    def _initialize_reference_geometry(
        self,
        *,
        scene_center_np: np.ndarray,
        current_position: np.ndarray,
    ) -> None:
        if self.reference_initialized:
            return
        self.reference_scene_center = np.asarray(
            scene_center_np, dtype=np.float64
        ).copy()
        self.reference_radius = float(
            np.linalg.norm(
                np.asarray(current_position, dtype=np.float64)
                - self.reference_scene_center
            )
        )
        self.reference_initialized = True

    def _build_cartesian_result(
        self,
        *,
        idx: int,
        viewpoint_source: str,
        current_c2w: np.ndarray,
        next_c2w: np.ndarray,
        desired_c2w: np.ndarray,
        reference_scene_center: np.ndarray,
        current_radius: float,
        radial_error: float,
        reference_radius: float,
        current_theta: float,
        current_phi: float,
        next_theta: float,
        next_phi: float,
        grad_theta: float,
        grad_phi: float,
        grad_norm: float,
        grad_theta_compressed: float,
        grad_phi_compressed: float,
        grad_norm_compressed: float,
        num_gaussians: int,
        fisher_score: float,
        scaled_spherical_velocity: np.ndarray,
        applied_spherical_velocity: np.ndarray,
        vt_world: np.ndarray,
        vn_world: np.ndarray,
        velocity_raw_world: np.ndarray,
        velocity_world: np.ndarray,
        rotvec_error: np.ndarray,
        angular_velocity_world: np.ndarray,
        angular_speed_raw: float,
        angular_speed_applied: float,
        next_position: np.ndarray,
        linear_speed_raw: float,
        linear_speed_applied: float,
        step_scale_theta: float,
        step_scale_phi: float,
        spherical_speed_min: float,
        clipped_spherical_speed: bool,
        clip_scale_ratio: float,
    ) -> MotionPolicyResult:
        """Pack a fully computed Cartesian-control step into MotionPolicyResult."""
        theta_rate_unclipped = float(scaled_spherical_velocity[0])
        phi_rate_unclipped = float(scaled_spherical_velocity[1])
        theta_rate = float(applied_spherical_velocity[0])
        phi_rate = float(applied_spherical_velocity[1])

        return MotionPolicyResult(
            idx=int(idx),
            viewpoint_source=viewpoint_source,
            controller_mode=self.controller_mode,
            cartesian=self.cartesian,
            current_c2w=current_c2w,
            next_c2w=next_c2w,
            scene_center=reference_scene_center.tolist(),
            reference_scene_center=reference_scene_center.tolist(),
            look_at_target=reference_scene_center.tolist(),
            desired_c2w=desired_c2w,
            radius=float(reference_radius),
            reference_radius=float(reference_radius),
            current_radius=float(current_radius),
            radial_error=radial_error,
            dt=self.dt,
            current_theta=current_theta,
            current_phi=current_phi,
            next_theta=next_theta,
            next_phi=next_phi,
            grad_theta_raw=grad_theta,
            grad_phi_raw=grad_phi,
            grad_norm_raw=grad_norm,
            grad_theta_compressed=float(grad_theta_compressed),
            grad_phi_compressed=float(grad_phi_compressed),
            grad_norm_compressed=float(grad_norm_compressed),
            num_gaussians=int(num_gaussians),
            fisher_score=fisher_score,
            scaled_theta=theta_rate_unclipped,
            scaled_phi=phi_rate_unclipped,
            delta_theta_applied=theta_rate,
            delta_phi_applied=phi_rate,
            velocity_raw_world=velocity_raw_world.astype(np.float64),
            vt_world=vt_world.astype(np.float64),
            vn_world=vn_world.astype(np.float64),
            velocity_world=velocity_world.astype(np.float64),
            rotvec_error=rotvec_error.astype(np.float64),
            angular_velocity_world=angular_velocity_world.astype(np.float64),
            angular_speed_raw=angular_speed_raw,
            angular_speed_applied=angular_speed_applied,
            angular_gain=float(self.angular_gain),
            enable_angular=bool(self.enable_angular),
            next_position=next_position.astype(np.float64).tolist(),
            step_scale_theta=float(step_scale_theta),
            step_scale_phi=float(step_scale_phi),
            linear_speed_limit=float(self.linear_vel_max),
            linear_speed_raw=linear_speed_raw,
            linear_speed_applied=linear_speed_applied,
            spherical_speed_limit=float("nan"),
            spherical_speed_min=float(spherical_speed_min),
            spherical_speed_raw=float(math.hypot(grad_theta, grad_phi)),
            spherical_speed_scaled=float(np.linalg.norm(scaled_spherical_velocity)),
            spherical_speed_applied=float(np.linalg.norm(applied_spherical_velocity)),
            clip_scale_ratio=float(clip_scale_ratio),
            speed_clipped=bool(clipped_spherical_speed),
            should_stop=False,
            stop_reason="normal_step",
            planner_output_mode=self.planner_output_mode,
        )

    @staticmethod
    def _build_w2c44_from_camera(camera: Any) -> np.ndarray:
        w2c = np.eye(4, dtype=np.float64)
        w2c[:3, :3] = camera.R.detach().float().cpu().numpy()
        w2c[:3, 3] = camera.T.detach().float().cpu().numpy()
        return w2c

    @staticmethod
    def _center_from_gaussians(gs_backend: Any) -> Optional[torch.Tensor]:
        xyz = getattr(gs_backend.gaussians, "get_xyz", None)
        if xyz is None:
            return None
        xyz = xyz.detach()
        if xyz.numel() == 0:
            return None
        xyz = xyz.reshape(-1, 3)
        finite_mask = torch.isfinite(xyz).all(dim=1)
        if not torch.any(finite_mask):
            return None
        return xyz[finite_mask].mean(dim=0).detach().float()

    @staticmethod
    def _count_gaussians(gs_backend: Any) -> int:
        gaussians = getattr(gs_backend, "gaussians", None)
        if gaussians is None:
            return 0
        xyz = getattr(gaussians, "get_xyz", None)
        if xyz is None:
            return 0
        if isinstance(xyz, torch.Tensor):
            if xyz.ndim == 0 or xyz.numel() == 0:
                return 0
            return int(xyz.shape[0])
        try:
            return int(len(xyz))
        except Exception:
            return 0

    @staticmethod
    def _compress_gradient_component(
        grad_component: float, num_gaussians: int
    ) -> float:
        # Use sign-preserving log1p(|grad| / num_gaussians) for stable scale normalization.
        grad_abs = abs(float(grad_component))
        if num_gaussians > 0:
            grad_abs = grad_abs / float(num_gaussians)
        return float(math.copysign(math.log1p(grad_abs), float(grad_component)))

    def _get_grad(
        self,
        *,
        gs_backend: Any,
        hemi_cam: Any,
    ) -> tuple[Any, float, float, dict[str, float]]:
        """Compute Fisher score and raw theta/phi gradient at the current hemisphere pose."""
        fisher_eval = gs_backend.fisher_eval
        history_t0 = time.perf_counter()
        cached_history = getattr(fisher_eval, "precomputed_history_stat", None)
        history_source = "missing"
        if cached_history is not None:
            history_source = "precomputed"
        elif getattr(gs_backend, "keyviewpoints", None):
            history_source = "recomputed"
        history_stat = fisher_eval.get_history_stat(gs_backend.keyviewpoints, prefer_cached=True)
        history_ms = (time.perf_counter() - history_t0) * 1000.0

        score_t0 = time.perf_counter()
        current_result = fisher_eval.compute_view_score(hemi_cam, history_stat)
        score_ms = (time.perf_counter() - score_t0) * 1000.0

        gradient_t0 = time.perf_counter()
        grad_theta_phi = fisher_eval.compute_view_gradient(
            hemi_cam,
            history_stat,
            eps=self.grad_eps,
        )
        gradient_ms = (time.perf_counter() - gradient_t0) * 1000.0
        grad_theta = float(grad_theta_phi[0].item())
        grad_phi = float(grad_theta_phi[1].item())
        timing = {
            "history_ms": float(history_ms),
            "score_ms": float(score_ms),
            "gradient_ms": float(gradient_ms),
            "history_source": str(history_source),
        }
        if self.verbose:
            self.logger.debug(
                "idx=%d fisher_timing history=%.2fms score=%.2fms gradient=%.2fms source=%s",
                int(getattr(hemi_cam, "uid", -1) if hasattr(hemi_cam, "uid") else -1),
                float(history_ms),
                float(score_ms),
                float(gradient_ms),
                str(history_source),
            )
        return current_result, grad_theta, grad_phi, timing

    def _cal_e_n(
        self,
        *,
        grad_theta: float,
        grad_phi: float,
        gaussian_count: int,
    ) -> tuple[float, float, float, np.ndarray, float, float, float]:
        """Compute desired tangential spherical velocity with Gaussian-count/log compression."""
        grad_theta_compressed = self._compress_gradient_component(
            grad_theta, gaussian_count
        )
        grad_phi_compressed = self._compress_gradient_component(
            grad_phi, gaussian_count
        )
        grad_norm_compressed = float(
            math.hypot(grad_theta_compressed, grad_phi_compressed)
        )
        effective_step_scale = (
            self.fisher_step_scale / float(gaussian_count)
            if gaussian_count > 0
            else self.fisher_step_scale
        )
        scaled_spherical_velocity = np.array(
            [
                self.fisher_step_scale * grad_theta_compressed,
                self.fisher_step_scale * grad_phi_compressed,
            ],
            dtype=np.float64,
        )
        spherical_speed_limit = float(
            math.hypot(self.max_delta_theta, self.max_delta_phi)
        )
        spherical_speed_min = float(self.spherical_speed_min)
        return (
            grad_theta_compressed,
            grad_phi_compressed,
            grad_norm_compressed,
            scaled_spherical_velocity,
            spherical_speed_limit,
            spherical_speed_min,
            effective_step_scale,
        )

    def _cal_e_t(
        self,
        *,
        current_position: np.ndarray,
        reference_scene_center: np.ndarray,
        current_radius: float,
        reference_radius: float,
    ) -> tuple[np.ndarray, float]:
        """Compute desired radial (normal) velocity for radius regulation."""
        radial_error = float(current_radius - reference_radius)
        radial_active = bool(abs(radial_error) > self.radial_deadband)
        if not radial_active:
            self._vn_cmd_preclip_norm = 0.0
            return np.zeros(3, dtype=np.float64), radial_error

        radial_offset = np.asarray(current_position, dtype=np.float64).reshape(
            3
        ) - np.asarray(reference_scene_center, dtype=np.float64).reshape(3)
        radial_gap = float(reference_radius - current_radius)
        denom = float(current_radius + 1e-6)
        if self.control_law_mode == "dt_consistent":
            n_hat = radial_offset / denom
            vn_world = (radial_gap / self.dt) * n_hat
        else:
            radial_correction = radial_offset * (radial_gap / denom)
            vn_world = self.radial_gain * radial_correction
        self._vn_cmd_preclip_norm = float(np.linalg.norm(vn_world))
        return vn_world, radial_error

    def _cal_omega(
        self,
        *,
        current_c2w: np.ndarray,
        current_position: np.ndarray,
        reference_scene_center: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
        """Compute desired world-frame angular velocity with gain and max-speed clipping."""
        # 1) 基于“相机当前位置 + 场景中心”构造期望朝向（保持看向场景中心）。
        desired_c2w = self._compute_desired_orientation(
            current_position=current_position,
            reference_scene_center=reference_scene_center,
            current_c2w=current_c2w,
        )
        # 2) 取当前/期望旋转矩阵并计算相对旋转误差 R_err = R_des * R_cur^T。
        current_rotation = np.asarray(current_c2w[:3, :3], dtype=np.float64)
        desired_rotation = np.asarray(desired_c2w[:3, :3], dtype=np.float64)
        # R_err 表示“从当前朝向旋到期望朝向”所需的增量旋转。
        rotation_error = desired_rotation @ current_rotation.T
        # 3) 将旋转误差映射为旋转向量（方向=旋转轴，模长=旋转角度）。
        rotvec_error = R.from_matrix(rotation_error).as_rotvec().astype(np.float64)

        # 原始角速度误差幅值（尚未乘增益、尚未限幅）。
        angular_speed_raw = float(np.linalg.norm(rotvec_error))
        # 4) 仅当角速度控制开启且误差超过死区时才输出角速度，避免小抖动。
        if self.enable_angular and angular_speed_raw > self.angular_speed_deadband:
            if self.control_law_mode == "dt_consistent":
                angular_velocity_world = rotvec_error / self.dt
            else:
                # 先按增益放大，再做最大角速度限幅。
                angular_velocity_world = self.angular_gain * rotvec_error
            angular_speed_unclipped = float(np.linalg.norm(angular_velocity_world))
            self._omega_cmd_preclip_norm = angular_speed_unclipped
            if angular_speed_unclipped > self.angular_speed_max:
                angular_velocity_world *= (
                    self.angular_speed_max / angular_speed_unclipped
                )
                angular_speed_applied = float(np.linalg.norm(angular_velocity_world))
                self.logger.warning(
                    (
                        "触发角速度限幅：raw=%.6f rad/s -> clipped=%.6f rad/s "
                        "(angular_speed_max=%.6f)"
                    ),
                    angular_speed_unclipped,
                    angular_speed_applied,
                    self.angular_speed_max,
                )
        else:
            # 控制关闭或误差太小：直接输出零角速度。
            angular_velocity_world = np.zeros(3, dtype=np.float64)
            self._omega_cmd_preclip_norm = 0.0

        # 最终实际下发的角速度幅值（经过死区和限幅后）。
        angular_speed_applied = float(np.linalg.norm(angular_velocity_world))
        return (
            desired_c2w,
            rotvec_error,
            angular_velocity_world,
            angular_speed_raw,
            angular_speed_applied,
        )

    def _cal_v_world(
        self,
        *,
        vt_world: np.ndarray,
        vn_world: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float, float]:
        """Compute desired world-frame linear velocity and apply max-speed clipping."""
        velocity_raw_world = np.asarray(vt_world, dtype=np.float64).reshape(
            3
        ) + np.asarray(vn_world, dtype=np.float64).reshape(3)
        linear_speed_raw = float(np.linalg.norm(velocity_raw_world))
        if linear_speed_raw <= 1e-9:
            velocity_world = np.zeros(3, dtype=np.float64)
            linear_speed_applied = 0.0
        else:
            linear_scale = min(linear_speed_raw, self.linear_vel_max) / linear_speed_raw
            velocity_world = velocity_raw_world * linear_scale
            linear_speed_applied = float(np.linalg.norm(velocity_world))
            if linear_speed_raw > self.linear_vel_max:
                self.logger.warning(
                    (
                        "触发线速度限幅：raw=%.6f m/s -> clipped=%.6f m/s "
                        "(linear_vel_max=%.6f)"
                    ),
                    linear_speed_raw,
                    linear_speed_applied,
                    self.linear_vel_max,
                )
        return (
            velocity_raw_world,
            velocity_world,
            linear_speed_raw,
            linear_speed_applied,
        )

    @staticmethod
    def _center_from_keyviews(gs_backend: Any) -> Optional[torch.Tensor]:
        keyviews = getattr(gs_backend, "keyviewpoints", None)
        if not keyviews:
            return None
        centers = []
        for viewpoint in keyviews:
            center = getattr(viewpoint, "camera_center", None)
            if center is None:
                continue
            center = center.detach().reshape(3)
            if torch.isfinite(center).all():
                centers.append(center.float())
        if not centers:
            return None
        return torch.stack(centers, dim=0).mean(dim=0)

    def _get_scene_center(self, gs_backend: Any) -> Optional[torch.Tensor]:
        if hasattr(gs_backend, "get_fisher_scene_center"):
            center = gs_backend.get_fisher_scene_center()
            if center is None:
                return None
            if isinstance(center, torch.Tensor):
                return center.detach().float()
            return torch.as_tensor(center, dtype=torch.float32)

        center = getattr(gs_backend, "sence_center", None)
        if center is None and hasattr(gs_backend, "tsdfs"):
            center = gs_backend.tsdfs.get_pointcloud_center()
            if center is not None:
                gs_backend.sence_center = center
        if center is None:
            center = self._center_from_gaussians(gs_backend)
            if center is not None and self.verbose:
                self.logger.debug(
                    "scene_center 回退：使用高斯均值 %s",
                    center.detach().cpu().numpy().tolist(),
                )
        if center is None:
            center = self._center_from_keyviews(gs_backend)
            if center is not None and self.verbose:
                self.logger.debug(
                    "scene_center 回退：使用关键帧相机均值 %s",
                    center.detach().cpu().numpy().tolist(),
                )
        if center is None:
            return None
        if isinstance(center, torch.Tensor):
            return center.detach().float()
        return torch.as_tensor(center, dtype=torch.float32)

    @staticmethod
    def _infer_runtime_device(gs_backend: Any) -> str:
        if hasattr(gs_backend, "get_runtime_device"):
            return str(gs_backend.get_runtime_device())
        if getattr(gs_backend, "keyviewpoints", None):
            return str(gs_backend.keyviewpoints[-1].device)
        try:
            params = gs_backend.gaussians.capture()
            for value in params:
                if isinstance(value, torch.Tensor):
                    return str(value.device)
        except Exception:
            pass
        return "cuda:0" if torch.cuda.is_available() else "cpu"

    def _build_tracking_camera_from_c2w(
        self,
        *,
        c2w: np.ndarray,
        intrinsics_vec: Sequence[float],
        image_size: Sequence[int],
        idx: int,
        device: str,
    ) -> Any:
        from ...utils.camera_utils import Camera
        from ...utils.graphics_utils import getProjectionMatrix2

        intrinsics = np.asarray(intrinsics_vec, dtype=np.float64).reshape(-1)
        if intrinsics.shape != (4,):
            raise ValueError(
                f"intrinsics_vec must have shape (4,), got {intrinsics.shape}"
            )
        if len(image_size) != 2:
            raise ValueError(f"image_size must be (height, width), got {image_size}")

        height = int(image_size[0])
        width = int(image_size[1])
        fx, fy, cx, cy = [float(v) for v in intrinsics.tolist()]
        color = torch.zeros((3, height, width), dtype=torch.float32, device=device)
        depth = torch.ones((height, width), dtype=torch.float32, device=device)
        w2c = torch.as_tensor(c2w_to_w2c(c2w), dtype=torch.float32, device=device)
        projection_matrix = getProjectionMatrix2(
            znear=0.01,
            zfar=100.0,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            W=width,
            H=height,
        ).transpose(0, 1)
        return Camera.init_from_tracking(
            color=color,
            depth=depth,
            pose=w2c,
            idx=int(idx),
            projection_matrix=projection_matrix,
            K=[fx, fy, cx, cy, width, height],
            tstamp=int(idx),
        )

    def _to_hemisphere_camera(
        self,
        *,
        gs_backend: Any,
        current_viewpoint: Any,
        scene_center: torch.Tensor,
        idx: int,
        intrinsics_vec: Sequence[float] | None = None,
        image_size: Sequence[int] | None = None,
    ) -> tuple[Any, str]:
        from ...utils.camera_utils import Camera, HemisphereCamera

        if isinstance(current_viewpoint, HemisphereCamera):
            return current_viewpoint.clone(), "hemisphere_camera"
        if isinstance(current_viewpoint, Camera):
            return HemisphereCamera.from_camera(
                current_viewpoint, scene_center
            ), "camera"

        c2w = np.asarray(current_viewpoint, dtype=np.float64)
        if c2w.shape != (4, 4):
            raise TypeError(
                "current_viewpoint must be a Camera/HemisphereCamera or a 4x4 c2w matrix"
            )
        if intrinsics_vec is None or image_size is None:
            raise ValueError(
                "intrinsics_vec and image_size are required when current_viewpoint is a c2w matrix"
            )
        temp_camera = self._build_tracking_camera_from_c2w(
            c2w=c2w,
            intrinsics_vec=intrinsics_vec,
            image_size=image_size,
            idx=idx,
            device=self._infer_runtime_device(gs_backend),
        )
        return HemisphereCamera.from_camera(temp_camera, scene_center), "c2w_matrix"

    def next_pose(
        self,
        *,
        gs_backend: Any,
        current_viewpoint: Any,
        idx: int,
        intrinsics_vec: Sequence[float] | None = None,
        image_size: Sequence[int] | None = None,
    ) -> MotionPolicyResult:
        """Compute the next pose from Fisher gradient, velocity decomposition, and safety clipping."""
        policy_t0 = time.perf_counter()
        fisher_ms = 0.0
        s2c_ms = 0.0
        # Resolve current pose/center context and keep a fixed reference sphere for Cartesian control.
        scene_center = self._get_scene_center(gs_backend)
        if scene_center is None:
            raise RuntimeError(
                "scene center is unavailable; cannot compute a Fisher-driven next pose"
            )

        hemi_cam, viewpoint_source = self._to_hemisphere_camera(
            gs_backend=gs_backend,
            current_viewpoint=current_viewpoint,
            scene_center=scene_center,
            idx=idx,
            intrinsics_vec=intrinsics_vec,
            image_size=image_size,
        )

        scene_center_np = scene_center.detach().cpu().numpy().astype(np.float64)
        radius = float(hemi_cam.radius)
        current_theta = float(hemi_cam.theta.item())
        current_phi = float(hemi_cam.phi.item())
        current_c2w = self._extract_current_c2w(
            current_viewpoint=current_viewpoint,
            viewpoint_source=viewpoint_source,
            current_theta=current_theta,
            current_phi=current_phi,
            scene_center_np=scene_center_np,
            radius=radius,
        )
        self._initialize_reference_geometry(
            scene_center_np=scene_center_np,
            current_position=current_c2w[:3, 3],
        )
        reference_scene_center = np.asarray(
            self.reference_scene_center, dtype=np.float64
        )
        reference_radius = float(self.reference_radius)
        control_scene_center = (
            torch.as_tensor(reference_scene_center, dtype=torch.float32)
            if self.cartesian
            else scene_center
        )
        if self.cartesian:
            hemi_cam, viewpoint_source = self._to_hemisphere_camera(
                gs_backend=gs_backend,
                current_viewpoint=current_viewpoint,
                scene_center=control_scene_center,
                idx=idx,
                intrinsics_vec=intrinsics_vec,
                image_size=image_size,
            )

        # 1) Evaluate Fisher and raw angular gradient at the current viewpoint.
        fisher_t0 = time.perf_counter()
        current_result, grad_theta, grad_phi, grad_timing = self._get_grad(
            gs_backend=gs_backend,
            hemi_cam=hemi_cam,
        )
        fisher_ms = (time.perf_counter() - fisher_t0) * 1000.0

        current_theta = float(hemi_cam.theta.item())
        current_phi = float(hemi_cam.phi.item())
        grad_norm = float(math.hypot(grad_theta, grad_phi))
        radius = float(hemi_cam.radius)
        gaussian_count = self._count_gaussians(gs_backend)

        # 2) Compute desired tangential velocity (includes /num_gaussians + log compression).
        (
            grad_theta_compressed,
            grad_phi_compressed,
            grad_norm_compressed,
            scaled_spherical_velocity,
            spherical_speed_limit,
            spherical_speed_min,
            effective_step_scale,
        ) = self._cal_e_n(
            grad_theta=grad_theta,
            grad_phi=grad_phi,
            gaussian_count=gaussian_count,
        )

        scaled_spherical_speed = float(np.linalg.norm(scaled_spherical_velocity))
        if scaled_spherical_speed < spherical_speed_min:
            current_radius = float(
                np.linalg.norm(
                    current_c2w[:3, 3].astype(np.float64) - reference_scene_center
                )
            )
            radial_error = float(current_radius - reference_radius)
            stop_result = self._build_stop_result(
                idx=idx,
                viewpoint_source=viewpoint_source,
                controller_mode=self.controller_mode,
                cartesian=self.cartesian,
                current_c2w=current_c2w,
                scene_center_np=scene_center_np,
                reference_scene_center=reference_scene_center,
                look_at_target=reference_scene_center,
                radius=radius,
                reference_radius=reference_radius,
                current_radius=current_radius,
                radial_error=radial_error,
                dt=self.dt,
                current_theta=current_theta,
                current_phi=current_phi,
                grad_theta=grad_theta,
                grad_phi=grad_phi,
                grad_norm=grad_norm,
                grad_theta_compressed=grad_theta_compressed,
                grad_phi_compressed=grad_phi_compressed,
                grad_norm_compressed=grad_norm_compressed,
                num_gaussians=gaussian_count,
                fisher_score=float(current_result.score),
                scaled_spherical_velocity=scaled_spherical_velocity,
                step_scale_theta=effective_step_scale,
                step_scale_phi=effective_step_scale,
                spherical_speed_min=spherical_speed_min,
                spherical_speed_limit=spherical_speed_limit,
                speed_clipped=False,
                clip_scale_ratio=1.0,
            )
            stop_result.planner_output_mode = self.planner_output_mode
            policy_total_ms = (time.perf_counter() - policy_t0) * 1000.0
            self.last_timing = {
                "history_ms": float(grad_timing.get("history_ms", 0.0)),
                "score_ms": float(grad_timing.get("score_ms", 0.0)),
                "gradient_ms": float(grad_timing.get("gradient_ms", 0.0)),
                "fisher_ms": float(fisher_ms),
                "s2c_ms": float(s2c_ms),
                "policy_total_ms": float(policy_total_ms),
                "history_source": str(grad_timing.get("history_source", "missing")),
            }
            if self.verbose:
                self.logger.debug(
                    "idx=%d mode=%s src=%s raw=(%.6f, %.6f) comp=(%.6f, %.6f) "
                    "scaled=(%.6f, %.6f) |u_scaled|=%.6f < min=%.6f -> stop",
                    stop_result.idx,
                    stop_result.controller_mode,
                    stop_result.viewpoint_source,
                    grad_theta,
                    grad_phi,
                    grad_theta_compressed,
                    grad_phi_compressed,
                    stop_result.scaled_theta,
                    stop_result.scaled_phi,
                    stop_result.spherical_speed_scaled,
                    stop_result.spherical_speed_min,
                )
            return stop_result

        if self.cartesian:
            applied_spherical_velocity = scaled_spherical_velocity.copy()
            clipped_spherical_speed = False
            clip_scale_ratio = 1.0
        else:
            applied_spherical_velocity, clipped_spherical_speed, clip_scale_ratio = (
                self._clip_vector_norm(scaled_spherical_velocity, spherical_speed_limit)
            )

        if self.cartesian:
            # 3) Cartesian branch: tangential speed -> radial correction -> linear speed clip.
            s2c_t0 = time.perf_counter()
            current_position = np.asarray(current_c2w[:3, 3], dtype=np.float64)
            current_radius, _, _ = _position_to_spherical(
                current_position, reference_scene_center
            )
            e_theta, e_phi, _ = _local_frame_from_theta_phi(current_theta, current_phi)
            delta_theta_unclipped = float(scaled_spherical_velocity[0])
            delta_phi_unclipped = float(scaled_spherical_velocity[1])
            delta_theta = float(applied_spherical_velocity[0])
            delta_phi = float(applied_spherical_velocity[1])
            if self.planner_output_mode == "spherical_delta":
                next_theta = self._wrap_theta(current_theta + delta_theta)
                next_phi = self._clamp_phi(current_phi + delta_phi)
                next_position = (
                    reference_scene_center
                    + current_radius * _spherical_direction(next_theta, next_phi)
                )
                desired_c2w = self._compute_desired_orientation(
                    current_position=current_position,
                    reference_scene_center=reference_scene_center,
                    current_c2w=current_c2w,
                )
                next_c2w = current_c2w.copy()
                next_c2w[:3, 3] = next_position
                next_c2w[:3, :3] = desired_c2w[:3, :3]
                radial_error = float(current_radius - reference_radius)
                s2c_ms = (time.perf_counter() - s2c_t0) * 1000.0
                result = MotionPolicyResult(
                    idx=int(idx),
                    viewpoint_source=viewpoint_source,
                    controller_mode=self.controller_mode,
                    cartesian=self.cartesian,
                    current_c2w=current_c2w,
                    next_c2w=next_c2w,
                    scene_center=reference_scene_center.tolist(),
                    reference_scene_center=reference_scene_center.tolist(),
                    look_at_target=reference_scene_center.tolist(),
                    desired_c2w=desired_c2w,
                    radius=float(reference_radius),
                    reference_radius=float(reference_radius),
                    current_radius=float(current_radius),
                    radial_error=radial_error,
                    dt=self.dt,
                    current_theta=current_theta,
                    current_phi=current_phi,
                    next_theta=next_theta,
                    next_phi=next_phi,
                    grad_theta_raw=grad_theta,
                    grad_phi_raw=grad_phi,
                    grad_norm_raw=grad_norm,
                    grad_theta_compressed=float(grad_theta_compressed),
                    grad_phi_compressed=float(grad_phi_compressed),
                    grad_norm_compressed=float(grad_norm_compressed),
                    num_gaussians=int(gaussian_count),
                    fisher_score=float(current_result.score),
                    scaled_theta=delta_theta_unclipped,
                    scaled_phi=delta_phi_unclipped,
                    delta_theta_applied=delta_theta,
                    delta_phi_applied=delta_phi,
                    velocity_raw_world=np.zeros(3, dtype=np.float64),
                    vt_world=np.zeros(3, dtype=np.float64),
                    vn_world=np.zeros(3, dtype=np.float64),
                    velocity_world=np.zeros(3, dtype=np.float64),
                    rotvec_error=np.zeros(3, dtype=np.float64),
                    angular_velocity_world=np.zeros(3, dtype=np.float64),
                    angular_speed_raw=0.0,
                    angular_speed_applied=0.0,
                    angular_gain=float(self.angular_gain),
                    enable_angular=bool(self.enable_angular),
                    next_position=next_position.astype(np.float64).tolist(),
                    step_scale_theta=float(effective_step_scale),
                    step_scale_phi=float(effective_step_scale),
                    linear_speed_limit=float(self.linear_vel_max),
                    linear_speed_raw=0.0,
                    linear_speed_applied=0.0,
                    spherical_speed_limit=float("nan"),
                    spherical_speed_min=float(spherical_speed_min),
                    spherical_speed_raw=float(math.hypot(grad_theta, grad_phi)),
                    spherical_speed_scaled=float(np.linalg.norm(scaled_spherical_velocity)),
                    spherical_speed_applied=float(np.linalg.norm(applied_spherical_velocity)),
                    clip_scale_ratio=float(clip_scale_ratio),
                    speed_clipped=bool(clipped_spherical_speed),
                    should_stop=False,
                    stop_reason="normal_step",
                    planner_output_mode="spherical_delta",
                )
                if self.verbose:
                    self.logger.debug(
                        "idx=%d mode=%s src=%s spherical_delta=(%.6f, %.6f) stop=%s",
                        result.idx,
                        result.controller_mode,
                        result.viewpoint_source,
                        result.delta_theta_applied,
                        result.delta_phi_applied,
                        result.should_stop,
                    )
                policy_total_ms = (time.perf_counter() - policy_t0) * 1000.0
                self.last_timing = {
                    "history_ms": float(grad_timing.get("history_ms", 0.0)),
                    "score_ms": float(grad_timing.get("score_ms", 0.0)),
                    "gradient_ms": float(grad_timing.get("gradient_ms", 0.0)),
                    "fisher_ms": float(fisher_ms),
                    "s2c_ms": float(s2c_ms),
                    "policy_total_ms": float(policy_total_ms),
                    "history_source": str(grad_timing.get("history_source", "missing")),
                }
                return result
            theta_rate = delta_theta / self.dt
            phi_rate = delta_phi / self.dt
            vt_world = current_radius * (theta_rate * e_theta + phi_rate * e_phi)
            vn_world, radial_error = self._cal_e_t(
                current_position=current_position,
                reference_scene_center=reference_scene_center,
                current_radius=current_radius,
                reference_radius=reference_radius,
            )
            (
                velocity_raw_world,
                velocity_world,
                linear_speed_raw,
                linear_speed_applied,
            ) = self._cal_v_world(
                vt_world=vt_world,
                vn_world=vn_world,
            )
            s2c_ms = (time.perf_counter() - s2c_t0) * 1000.0

            next_position = current_position + self.dt * velocity_world
            next_radius, next_theta, next_phi_raw = _position_to_spherical(
                next_position, reference_scene_center
            )
            next_phi = self._clamp_phi(next_phi_raw)
            if not math.isclose(next_phi, next_phi_raw, rel_tol=1e-9, abs_tol=1e-12):
                next_position = (
                    reference_scene_center
                    + next_radius * _spherical_direction(next_theta, next_phi)
                )
                _, next_theta, next_phi = _position_to_spherical(
                    next_position, reference_scene_center
                )

            # 4) Compute orientation command and clip angular speed.
            (
                desired_c2w,
                rotvec_error,
                angular_velocity_world,
                angular_speed_raw,
                angular_speed_applied,
            ) = self._cal_omega(
                current_c2w=current_c2w,
                current_position=current_position,
                reference_scene_center=reference_scene_center,
            )
            next_rotation = self._integrate_world_angular_velocity(
                current_rotation=np.asarray(current_c2w[:3, :3], dtype=np.float64),
                angular_velocity_world=angular_velocity_world,
                dt=self.dt,
            )
            next_c2w = np.eye(4, dtype=np.float64)
            next_c2w[:3, :3] = next_rotation
            next_c2w[:3, 3] = next_position

            result = self._build_cartesian_result(
                idx=int(idx),
                viewpoint_source=viewpoint_source,
                current_c2w=current_c2w,
                next_c2w=next_c2w,
                desired_c2w=desired_c2w,
                reference_scene_center=reference_scene_center,
                current_radius=float(current_radius),
                radial_error=radial_error,
                reference_radius=float(reference_radius),
                current_theta=current_theta,
                current_phi=current_phi,
                next_theta=next_theta,
                next_phi=next_phi,
                grad_theta=grad_theta,
                grad_phi=grad_phi,
                grad_norm=grad_norm,
                grad_theta_compressed=float(grad_theta_compressed),
                grad_phi_compressed=float(grad_phi_compressed),
                grad_norm_compressed=float(grad_norm_compressed),
                num_gaussians=int(gaussian_count),
                fisher_score=float(current_result.score),
                scaled_spherical_velocity=scaled_spherical_velocity,
                applied_spherical_velocity=applied_spherical_velocity,
                vt_world=vt_world.astype(np.float64),
                vn_world=vn_world.astype(np.float64),
                velocity_raw_world=velocity_raw_world.astype(np.float64),
                velocity_world=velocity_world.astype(np.float64),
                rotvec_error=rotvec_error.astype(np.float64),
                angular_velocity_world=angular_velocity_world.astype(np.float64),
                angular_speed_raw=angular_speed_raw,
                angular_speed_applied=angular_speed_applied,
                next_position=next_position.astype(np.float64),
                linear_speed_raw=linear_speed_raw,
                linear_speed_applied=linear_speed_applied,
                step_scale_theta=float(effective_step_scale),
                step_scale_phi=float(effective_step_scale),
                spherical_speed_min=float(spherical_speed_min),
                clipped_spherical_speed=bool(clipped_spherical_speed),
                clip_scale_ratio=float(clip_scale_ratio),
            )
            if self.verbose:
                self.logger.debug(
                    "idx=%d mode=%s src=%s r=%.4f raw=(%.6f, %.6f) comp=(%.6f, %.6f) "
                    "scaled=(%.6f, %.6f) clipped=(%.6f, %.6f) |u_raw|=%.6f |u_scaled|=%.6f "
                    "dr=%.6f |vt|=%.6f |vn|=%.6f |v_raw|=%.6f |v|=%.6f/%.6f "
                    "|rotvec_err|=%.6f |omega|=%.6f ctrl=%s |omega_preclip|=%.6f "
                    "|vn_preclip|=%.6f stop=%s",
                    result.idx,
                    result.controller_mode,
                    result.viewpoint_source,
                    result.current_radius,
                    grad_theta,
                    grad_phi,
                    grad_theta_compressed,
                    grad_phi_compressed,
                    result.scaled_theta,
                    result.scaled_phi,
                    result.delta_theta_applied,
                    result.delta_phi_applied,
                    result.spherical_speed_raw,
                    result.spherical_speed_scaled,
                    result.radial_error,
                    np.linalg.norm(result.vt_world),
                    np.linalg.norm(result.vn_world),
                    result.linear_speed_raw,
                    result.linear_speed_applied,
                    result.linear_speed_limit,
                    result.angular_speed_raw,
                    result.angular_speed_applied,
                    self.control_law_mode,
                    self._omega_cmd_preclip_norm,
                    self._vn_cmd_preclip_norm,
                    result.should_stop,
                )
            policy_total_ms = (time.perf_counter() - policy_t0) * 1000.0
            self.last_timing = {
                "history_ms": float(grad_timing.get("history_ms", 0.0)),
                "score_ms": float(grad_timing.get("score_ms", 0.0)),
                "gradient_ms": float(grad_timing.get("gradient_ms", 0.0)),
                "fisher_ms": float(fisher_ms),
                "s2c_ms": float(s2c_ms),
                "policy_total_ms": float(policy_total_ms),
                "history_source": str(grad_timing.get("history_source", "missing")),
            }
            return result

        delta_theta = float(applied_spherical_velocity[0])
        delta_phi = float(applied_spherical_velocity[1])
        next_theta = self._wrap_theta(current_theta + delta_theta)
        next_phi = self._clamp_phi(current_phi + delta_phi)
        next_c2w = _spherical_c2w(
            scene_center=scene_center_np,
            radius=radius,
            theta=next_theta,
            phi=next_phi,
        )

        result = MotionPolicyResult(
            idx=int(idx),
            viewpoint_source=viewpoint_source,
            controller_mode=self.controller_mode,
            cartesian=self.cartesian,
            current_c2w=current_c2w,
            next_c2w=next_c2w,
            scene_center=scene_center_np.tolist(),
            reference_scene_center=reference_scene_center.tolist(),
            look_at_target=reference_scene_center.tolist(),
            desired_c2w=next_c2w.copy(),
            radius=radius,
            reference_radius=reference_radius,
            current_radius=radius,
            radial_error=float(radius - reference_radius),
            dt=self.dt,
            current_theta=current_theta,
            current_phi=current_phi,
            next_theta=next_theta,
            next_phi=next_phi,
            grad_theta_raw=grad_theta,
            grad_phi_raw=grad_phi,
            grad_norm_raw=grad_norm,
            grad_theta_compressed=grad_theta_compressed,
            grad_phi_compressed=grad_phi_compressed,
            grad_norm_compressed=grad_norm_compressed,
            num_gaussians=gaussian_count,
            fisher_score=float(current_result.score),
            scaled_theta=float(scaled_spherical_velocity[0]),
            scaled_phi=float(scaled_spherical_velocity[1]),
            delta_theta_applied=delta_theta,
            delta_phi_applied=delta_phi,
            velocity_raw_world=np.zeros(3, dtype=np.float64),
            vt_world=np.zeros(3, dtype=np.float64),
            vn_world=np.zeros(3, dtype=np.float64),
            velocity_world=np.zeros(3, dtype=np.float64),
            rotvec_error=np.zeros(3, dtype=np.float64),
            angular_velocity_world=np.zeros(3, dtype=np.float64),
            angular_speed_raw=0.0,
            angular_speed_applied=0.0,
            angular_gain=0.0,
            enable_angular=False,
            next_position=next_c2w[:3, 3].astype(np.float64).tolist(),
            step_scale_theta=effective_step_scale,
            step_scale_phi=effective_step_scale,
            linear_speed_limit=0.0,
            linear_speed_raw=0.0,
            linear_speed_applied=0.0,
            spherical_speed_limit=(
                float("nan") if self.cartesian else spherical_speed_limit
            ),
            spherical_speed_min=spherical_speed_min,
            spherical_speed_raw=float(math.hypot(grad_theta, grad_phi)),
            spherical_speed_scaled=float(np.linalg.norm(scaled_spherical_velocity)),
            spherical_speed_applied=float(np.linalg.norm(applied_spherical_velocity)),
            clip_scale_ratio=float(clip_scale_ratio),
            speed_clipped=bool(clipped_spherical_speed),
            should_stop=False,
            stop_reason="normal_step",
            planner_output_mode=self.planner_output_mode,
        )

        if self.verbose:
            self.logger.debug(
                "idx=%d src=%s theta=%.4f->%.4f phi=%.4f->%.4f raw=(%.6f, %.6f) "
                "comp=(%.6f, %.6f) scaled=(%.6f, %.6f) clipped=(%.6f, %.6f) "
                "|u_raw|=%.6f |u_scaled|=%.6f |u_clip|=%.6f/%.6f score=%.6f clip=%s stop=%s",
                result.idx,
                result.viewpoint_source,
                result.current_theta,
                result.next_theta,
                result.current_phi,
                result.next_phi,
                result.grad_theta_raw,
                result.grad_phi_raw,
                result.grad_theta_compressed,
                result.grad_phi_compressed,
                result.scaled_theta,
                result.scaled_phi,
                result.delta_theta_applied,
                result.delta_phi_applied,
                result.spherical_speed_raw,
                result.spherical_speed_scaled,
                result.spherical_speed_applied,
                result.spherical_speed_limit,
                result.fisher_score,
                result.speed_clipped,
                result.should_stop,
            )
        policy_total_ms = (time.perf_counter() - policy_t0) * 1000.0
        self.last_timing = {
            "history_ms": float(grad_timing.get("history_ms", 0.0)),
            "score_ms": float(grad_timing.get("score_ms", 0.0)),
            "gradient_ms": float(grad_timing.get("gradient_ms", 0.0)),
            "fisher_ms": float(fisher_ms),
            "s2c_ms": float(s2c_ms),
            "policy_total_ms": float(policy_total_ms),
            "history_source": str(grad_timing.get("history_source", "missing")),
        }
        return result

    def next_pose_from_c2w(
        self,
        *,
        gs_backend: Any,
        current_c2w: np.ndarray,
        intrinsics_vec: Sequence[float],
        image_size: Sequence[int],
        idx: int,
    ) -> MotionPolicyResult:
        return self.next_pose(
            gs_backend=gs_backend,
            current_viewpoint=np.asarray(current_c2w, dtype=np.float64),
            idx=idx,
            intrinsics_vec=intrinsics_vec,
            image_size=image_size,
        )
