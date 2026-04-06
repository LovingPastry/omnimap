"""Phase-3 Fisher motion policy for the simulation loop.

This module keeps the policy logic on the simulation side:
- read the current OmniMap / GS state
- convert the current view to a hemisphere parameterization
- query the existing Fisher evaluator for angular gradients
- advance the camera on the upper hemisphere while keeping it looking at the scene

The controller intentionally preserves the Fisher gradient direction.
It first builds a 2D velocity in spherical coordinates `(theta, phi)`,
then applies a two-sided speed policy on that vector norm:
- if the norm is too large, shrink it while preserving direction
- if the norm is too small, treat the controller as converged and stop
Only afterwards is the spherical velocity mapped to either angular
updates or Cartesian tangent velocity.
"""

from __future__ import annotations

import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from .pose_utils import c2w_to_w2c, w2c_to_c2w


def _ensure_omnimap_import_paths() -> Path:
    """Make OmniMap's sibling-import layout resolvable from the simulation side."""
    repo_root = Path(__file__).resolve().parent.parent
    omnimap_dir = repo_root / "omnimap"
    for path in (repo_root, omnimap_dir):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
    return repo_root


def _sim_look_at_c2w(
    eye: np.ndarray,
    target: np.ndarray,
    up: np.ndarray | None = None,
) -> np.ndarray:
    """Build a sim-render-compatible `c2w` pose with camera +Y pointing image-down."""
    eye = np.asarray(eye, dtype=np.float64).reshape(3)
    target = np.asarray(target, dtype=np.float64).reshape(3)
    up = (
        np.array([0.0, 0.0, 1.0], dtype=np.float64)
        if up is None
        else np.asarray(up, dtype=np.float64).reshape(3)
    )

    forward = target - eye
    forward_norm = np.linalg.norm(forward)
    if forward_norm < 1e-12:
        raise ValueError("eye and target are too close; cannot build look-at pose")
    forward = forward / forward_norm

    right = np.cross(forward, up)
    right_norm = np.linalg.norm(right)
    if right_norm < 1e-12:
        fallback_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        right = np.cross(forward, fallback_up)
        right_norm = np.linalg.norm(right)
        if right_norm < 1e-12:
            raise ValueError("failed to construct a valid right axis for look-at pose")
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


def _sim_spherical_c2w(
    *,
    scene_center: np.ndarray,
    radius: float,
    theta: float,
    phi: float,
) -> np.ndarray:
    """Convert spherical state into the SceneSimulator-compatible camera convention."""
    center = np.asarray(scene_center, dtype=np.float64).reshape(3)
    eye = center + np.array(
        [
            radius * math.cos(phi) * math.cos(theta),
            radius * math.cos(phi) * math.sin(theta),
            radius * math.sin(phi),
        ],
        dtype=np.float64,
    )
    return _sim_look_at_c2w(eye=eye, target=center)


def _spherical_direction(
    theta: float,
    phi: float,
) -> np.ndarray:
    """Return the unit direction on the upper hemisphere for `(theta, phi)`."""
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
    """Return `(e_theta, e_phi, n_hat)` at one spherical state."""
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
    """Convert a 3D position back into `(radius, theta, phi)` around `scene_center`."""
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
    """Structured record of one Fisher-driven pose update."""

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

    def to_jsonable(self) -> dict[str, object]:
        data = asdict(self)
        for key, value in list(data.items()):
            if isinstance(value, np.ndarray):
                data[key] = value.tolist()
        return data

    @property
    def grad_theta(self) -> float:
        """Backward-compatible alias for existing debug prints."""
        return self.grad_theta_raw

    @property
    def grad_phi(self) -> float:
        """Backward-compatible alias for existing debug prints."""
        return self.grad_phi_raw

    @property
    def grad_norm(self) -> float:
        """Backward-compatible alias for existing debug prints."""
        return self.grad_norm_raw


class FisherMotionPolicy:
    """Advance the camera pose on the Fisher hemisphere using angle gradients."""

    def __init__(
        self,
        step_gain_theta: float = 0.03,
        step_gain_phi: float = 0.03,
        *,
        cartesian: bool = False,
        dt: float = 0.1,
        radial_gain: float = 2.0,
        linear_vel_max: float = 0.5,
        angular_gain: float = 2.0,
        enable_angular: bool = True,
        grad_eps: float = 0.01,
        spherical_speed_min: float = 1e-4,
        max_delta_theta: float = 0.20,
        max_delta_phi: float = 0.15,
        phi_min: float = 1e-3,
        phi_max: float | None = None,
        verbose: bool = True,
    ) -> None:
        if not np.isfinite(step_gain_theta) or not np.isfinite(step_gain_phi):
            raise ValueError("step gains must be finite")
        if not math.isclose(
            step_gain_theta, step_gain_phi, rel_tol=1e-12, abs_tol=1e-12
        ):
            raise ValueError(
                "use one shared fisher_step_scale; theta/phi gains must match"
            )
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
        if grad_eps <= 0:
            raise ValueError(f"grad_eps must be positive, got {grad_eps}")
        if spherical_speed_min < 0 or not np.isfinite(spherical_speed_min):
            raise ValueError(
                f"spherical_speed_min must be non-negative and finite, got {spherical_speed_min}"
            )
        if max_delta_theta <= 0 or max_delta_phi <= 0:
            raise ValueError("max deltas must be positive")

        self.step_gain_theta = float(step_gain_theta)
        self.step_gain_phi = float(step_gain_phi)
        self.controller_mode = "cartesian" if cartesian else "angular"
        self.cartesian = bool(cartesian)
        self.dt = float(dt)
        self.radial_gain = float(radial_gain)
        self.linear_vel_max = float(linear_vel_max)
        self.angular_gain = float(angular_gain)
        self.enable_angular = bool(enable_angular)
        self.grad_eps = float(grad_eps)
        self.spherical_speed_min = float(spherical_speed_min)
        self.max_delta_theta = float(max_delta_theta)
        self.max_delta_phi = float(max_delta_phi)
        self.phi_min = float(phi_min)
        self.phi_max = float(math.pi / 2.0 - 1e-3 if phi_max is None else phi_max)
        self.verbose = bool(verbose)
        self.reference_scene_center: Optional[np.ndarray] = None
        self.reference_radius: Optional[float] = None
        self.reference_initialized = False

        if not (0.0 <= self.phi_min <= self.phi_max <= math.pi / 2.0):
            raise ValueError(
                f"invalid phi range: phi_min={self.phi_min}, phi_max={self.phi_max}"
            )

    @staticmethod
    def _wrap_theta(theta: float) -> float:
        return float(theta % (2.0 * math.pi))

    def _clamp_phi(self, phi: float) -> float:
        """Keep the elevation on the valid upper-hemisphere interval."""
        return float(np.clip(phi, self.phi_min, self.phi_max))

    @staticmethod
    def _clip_vector_norm(
        vec: np.ndarray,
        limit: float,
    ) -> tuple[np.ndarray, bool, float]:
        """Clip a 2D spherical velocity by norm while preserving its direction."""
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
        """Integrate a world-frame angular velocity over one explicit Euler step on SO(3)."""
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
    ) -> np.ndarray:
        """Construct the desired camera orientation that points toward the reference sphere center."""
        return _sim_look_at_c2w(
            eye=np.asarray(current_position, dtype=np.float64),
            target=np.asarray(reference_scene_center, dtype=np.float64),
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
        fisher_score: float,
        scaled_spherical_velocity: np.ndarray,
        step_scale_theta: float,
        step_scale_phi: float,
        spherical_speed_min: float,
        spherical_speed_limit: float,
        speed_clipped: bool,
        clip_scale_ratio: float,
    ) -> "MotionPolicyResult":
        """Return a no-motion result when the spherical speed is below the stop threshold."""
        return MotionPolicyResult(
            idx=int(idx),
            viewpoint_source=viewpoint_source,
            controller_mode=controller_mode,
            cartesian=cartesian,
            current_c2w=current_c2w,
            next_c2w=current_c2w.copy(),
            scene_center=np.asarray(scene_center_np, dtype=np.float64).tolist(),
            reference_scene_center=np.asarray(reference_scene_center, dtype=np.float64).tolist(),
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
        """Resolve the authoritative current pose in sim rendering convention."""
        if viewpoint_source == "c2w_matrix":
            c2w = np.asarray(current_viewpoint, dtype=np.float64)
            if c2w.shape != (4, 4):
                raise ValueError(f"expected 4x4 c2w matrix, got {c2w.shape}")
            return c2w
        if viewpoint_source in {"camera", "hemisphere_camera"}:
            w2c = FisherMotionPolicy._build_w2c44_from_camera(current_viewpoint)
            return w2c_to_c2w(w2c)
        return _sim_spherical_c2w(
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
        """Lock the reference sphere once per policy lifecycle."""
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
        reference_scene_center: np.ndarray,
        reference_radius: float,
        current_theta: float,
        current_phi: float,
        grad_theta: float,
        grad_phi: float,
        grad_norm: float,
        fisher_score: float,
        scaled_spherical_velocity: np.ndarray,
        applied_spherical_velocity: np.ndarray,
        spherical_speed_min: float,
        spherical_speed_limit: float,
        clipped_spherical_speed: bool,
        clip_scale_ratio: float,
    ) -> MotionPolicyResult:
        """Advance one step using tangent + radial velocity in world coordinates."""
        current_position = np.asarray(current_c2w[:3, 3], dtype=np.float64)
        current_radius, _, _ = _position_to_spherical(
            current_position, reference_scene_center
        )
        radial_error = float(current_radius - reference_radius)

        e_theta, e_phi, n_hat = _local_frame_from_theta_phi(current_theta, current_phi)
        theta_rate_unclipped = float(scaled_spherical_velocity[0])
        phi_rate_unclipped = float(scaled_spherical_velocity[1])
        theta_rate = float(applied_spherical_velocity[0])
        phi_rate = float(applied_spherical_velocity[1])

        # Tangential command from Fisher policy in Cartesian coordinates.
        vt_world = current_radius * (theta_rate * e_theta + phi_rate * e_phi)

        # Radial correction follows velocity_cmd_algorithm.md:
        # e_n = (p-c) * (R-r) / (r+eps), and only contributes when r < R.
        radial_offset = current_position - reference_scene_center
        denom = float(current_radius + 1e-6)
        radial_gap = float(reference_radius - current_radius)
        radial_correction = radial_offset * (radial_gap / denom)
        radial_active = bool(current_radius < reference_radius)
        vn_world = self.radial_gain * radial_correction if radial_active else np.zeros(3, dtype=np.float64)

        velocity_raw_world = vt_world + vn_world
        linear_speed_raw = float(np.linalg.norm(velocity_raw_world))
        if linear_speed_raw <= 1e-9:
            velocity_world = np.zeros(3, dtype=np.float64)
            linear_speed_applied = 0.0
        else:
            linear_scale = min(linear_speed_raw, self.linear_vel_max) / linear_speed_raw
            velocity_world = velocity_raw_world * linear_scale
            linear_speed_applied = float(np.linalg.norm(velocity_world))
        next_position = current_position + self.dt * velocity_world
        next_radius, next_theta, next_phi_raw = _position_to_spherical(
            next_position, reference_scene_center
        )
        next_phi = self._clamp_phi(next_phi_raw)
        if not math.isclose(next_phi, next_phi_raw, rel_tol=1e-9, abs_tol=1e-12):
            next_position = reference_scene_center + next_radius * _spherical_direction(
                next_theta, next_phi
            )
            next_radius, next_theta, next_phi = _position_to_spherical(
                next_position, reference_scene_center
            )

        desired_c2w = self._compute_desired_orientation(
            current_position=current_position,
            reference_scene_center=reference_scene_center,
        )
        current_rotation = np.asarray(current_c2w[:3, :3], dtype=np.float64)
        desired_rotation = np.asarray(desired_c2w[:3, :3], dtype=np.float64)
        rotation_error = desired_rotation @ current_rotation.T
        rotvec_error = R.from_matrix(rotation_error).as_rotvec().astype(np.float64)
        angular_speed_raw = float(np.linalg.norm(rotvec_error))
        if self.enable_angular:
            angular_velocity_world = self.angular_gain * rotvec_error
        else:
            angular_velocity_world = np.zeros(3, dtype=np.float64)
        angular_speed_applied = float(np.linalg.norm(angular_velocity_world))
        next_rotation = self._integrate_world_angular_velocity(
            current_rotation=current_rotation,
            angular_velocity_world=angular_velocity_world,
            dt=self.dt,
        )
        next_c2w = np.eye(4, dtype=np.float64)
        next_c2w[:3, :3] = next_rotation
        next_c2w[:3, 3] = next_position

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
            step_scale_theta=self.step_gain_theta,
            step_scale_phi=self.step_gain_phi,
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
        )

    @staticmethod
    def _build_w2c44_from_camera(camera: Any) -> np.ndarray:
        """Extract a dense `w2c` matrix from an OmniMap camera object."""
        w2c = np.eye(4, dtype=np.float64)
        w2c[:3, :3] = camera.R.detach().float().cpu().numpy()
        w2c[:3, 3] = camera.T.detach().float().cpu().numpy()
        return w2c

    @staticmethod
    def _center_from_gaussians(gs_backend: Any) -> Optional[torch.Tensor]:
        """Fallback scene center estimate derived from the active Gaussian map."""
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
    def _center_from_keyviews(gs_backend: Any) -> Optional[torch.Tensor]:
        """Secondary fallback center estimate derived from keyframe camera centers."""
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
        """Resolve the best currently-available scene center for hemisphere motion."""
        center = getattr(gs_backend, "sence_center", None)
        if center is None and hasattr(gs_backend, "tsdfs"):
            center = gs_backend.tsdfs.get_pointcloud_center()
            if center is not None:
                gs_backend.sence_center = center
        if center is None:
            center = self._center_from_gaussians(gs_backend)
            if center is not None:
                gs_backend.sence_center = center
                if self.verbose:
                    print(
                        "[FisherMotionPolicy] scene_center fallback: using Gaussian mean "
                        f"{center.detach().cpu().numpy().tolist()}"
                    )
        if center is None:
            center = self._center_from_keyviews(gs_backend)
            if center is not None:
                gs_backend.sence_center = center
                if self.verbose:
                    print(
                        "[FisherMotionPolicy] scene_center fallback: using keyframe camera mean "
                        f"{center.detach().cpu().numpy().tolist()}"
                    )
        if center is None:
            return None
        if isinstance(center, torch.Tensor):
            return center.detach().float()
        return torch.as_tensor(center, dtype=torch.float32)

    @staticmethod
    def _infer_runtime_device(gs_backend: Any) -> str:
        """Infer which torch device OmniMap is currently using."""
        if getattr(gs_backend, "keyviewpoints", None):
            device = gs_backend.keyviewpoints[-1].device
            return str(device)
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
        """Wrap a raw `c2w` pose as an OmniMap `Camera` for Fisher queries."""
        _ensure_omnimap_import_paths()
        from gaussian.utils.camera_utils import Camera
        from gaussian.utils.graphics_utils import getProjectionMatrix2

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
        """Convert the current viewpoint into a `HemisphereCamera` consistently."""
        _ensure_omnimap_import_paths()
        from gaussian.utils.camera_utils import Camera, HemisphereCamera

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
        """使用 Fisher 角度梯度在半球面上前进一步。

        这是核心的第3/4阶段策略：
        1. 恢复 Fisher 半球上的当前视图
        2. 查询 `dF/dtheta` 和 `dF/dphi` (Fisher信息矩阵相对于球坐标角度的梯度)
        3. 缩放球面速度并应用双阈值策略
        4. 若球坐标速度模长过小，则直接视为收敛并停止
        5. 否则将更新后的半球状态转换回 `next_c2w` (下一相机位姿)
        """
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

        # Reuse OmniMap's existing information-gain logic directly so the simulator
        # only owns the motion decision, not a forked Fisher implementation.
        fisher_eval = gs_backend.fisher_eval
        history_stat = fisher_eval.compute_history_stat(gs_backend.keyviewpoints)
        current_result = fisher_eval.compute_view_score(hemi_cam, history_stat)
        grad_theta_phi = fisher_eval.compute_view_gradient(
            hemi_cam,
            history_stat,
            eps=self.grad_eps,
        )

        current_theta = float(hemi_cam.theta.item())
        current_phi = float(hemi_cam.phi.item())
        grad_theta = float(grad_theta_phi[0].item())
        grad_phi = float(grad_theta_phi[1].item())
        grad_norm = float(math.hypot(grad_theta, grad_phi))
        radius = float(hemi_cam.radius)

        # First map raw Fisher gradients into a 2D spherical velocity, then apply
        # a two-sided norm policy so the direction remains aligned with the gradient field.
        scaled_spherical_velocity = np.array(
            [
                self.step_gain_theta * grad_theta,
                self.step_gain_phi * grad_phi,
            ],
            dtype=np.float64,
        )

        spherical_speed_limit = float(
            math.hypot(self.max_delta_theta, self.max_delta_phi)
        )
        spherical_speed_min = float(self.spherical_speed_min)
        if self.cartesian:
            scaled_spherical_velocity = scaled_spherical_velocity / self.dt
            spherical_speed_min = spherical_speed_min / self.dt

        scaled_spherical_speed = float(np.linalg.norm(scaled_spherical_velocity))
        if scaled_spherical_speed < spherical_speed_min:
            current_radius = float(
                np.linalg.norm(current_c2w[:3, 3].astype(np.float64) - reference_scene_center)
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
                fisher_score=float(current_result.score),
                scaled_spherical_velocity=scaled_spherical_velocity,
                step_scale_theta=self.step_gain_theta,
                step_scale_phi=self.step_gain_phi,
                spherical_speed_min=spherical_speed_min,
                spherical_speed_limit=spherical_speed_limit,
                speed_clipped=False,
                clip_scale_ratio=1.0,
            )
            if self.verbose:
                print(
                    f"[FisherMotionPolicy] idx={stop_result.idx} mode={stop_result.controller_mode} "
                    f"src={stop_result.viewpoint_source} raw=({grad_theta:.6f}, {grad_phi:.6f}) "
                    f"scaled=({stop_result.scaled_theta:.6f}, {stop_result.scaled_phi:.6f}) "
                    f"|u_scaled|={stop_result.spherical_speed_scaled:.6f} "
                    f"< min={stop_result.spherical_speed_min:.6f} -> stop"
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
            result = self._build_cartesian_result(
                idx=idx,
                viewpoint_source=viewpoint_source,
                current_c2w=current_c2w,
                reference_scene_center=reference_scene_center,
                reference_radius=reference_radius,
                current_theta=current_theta,
                current_phi=current_phi,
                grad_theta=grad_theta,
                grad_phi=grad_phi,
                grad_norm=grad_norm,
                fisher_score=float(current_result.score),
                scaled_spherical_velocity=scaled_spherical_velocity,
                applied_spherical_velocity=applied_spherical_velocity,
                spherical_speed_min=spherical_speed_min,
                spherical_speed_limit=spherical_speed_limit,
                clipped_spherical_speed=clipped_spherical_speed,
                clip_scale_ratio=clip_scale_ratio,
            )
            if self.verbose:
                print(
                    f"[FisherMotionPolicy] idx={result.idx} mode={result.controller_mode} "
                    f"src={result.viewpoint_source} r={result.current_radius:.4f} "
                    f"raw=({grad_theta:.6f}, {grad_phi:.6f}) "
                    f"scaled=({result.scaled_theta:.6f}, {result.scaled_phi:.6f}) "
                    f"clipped=({result.delta_theta_applied:.6f}, {result.delta_phi_applied:.6f}) "
                    f"|u_raw|={result.spherical_speed_raw:.6f} "
                    f"|u_scaled|={result.spherical_speed_scaled:.6f} "
                    f"dr={result.radial_error:.6f} |vt|={np.linalg.norm(result.vt_world):.6f} "
                    f"|vn|={np.linalg.norm(result.vn_world):.6f} "
                    f"|v_raw|={result.linear_speed_raw:.6f} "
                    f"|v|={result.linear_speed_applied:.6f}/{result.linear_speed_limit:.6f} "
                    f"|rotvec_err|={result.angular_speed_raw:.6f} "
                    f"|omega|={result.angular_speed_applied:.6f} "
                    f"stop={result.should_stop}"
                )
            return result

        # Then update the spherical state and convert it back into the loop's authoritative pose.
        delta_theta = float(applied_spherical_velocity[0])
        delta_phi = float(applied_spherical_velocity[1])
        next_theta = self._wrap_theta(current_theta + delta_theta)
        next_phi = self._clamp_phi(current_phi + delta_phi)
        # Rebuild poses in the simulator's rendering convention so closed-loop
        # RGBD renders keep the same upright image orientation as Phase 1.
        next_c2w = _sim_spherical_c2w(
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
            step_scale_theta=self.step_gain_theta,
            step_scale_phi=self.step_gain_phi,
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
        )

        if self.verbose:
            print(
                f"[FisherMotionPolicy] idx={result.idx} src={result.viewpoint_source} "
                f"theta={result.current_theta:.4f}->{result.next_theta:.4f} "
                f"phi={result.current_phi:.4f}->{result.next_phi:.4f} "
                f"raw=({result.grad_theta_raw:.6f}, {result.grad_phi_raw:.6f}) "
                f"scaled=({result.scaled_theta:.6f}, {result.scaled_phi:.6f}) "
                f"clipped=({result.delta_theta_applied:.6f}, {result.delta_phi_applied:.6f}) "
                f"|u_raw|={result.spherical_speed_raw:.6f} "
                f"|u_scaled|={result.spherical_speed_scaled:.6f} "
                f"|u_clip|={result.spherical_speed_applied:.6f}/{result.spherical_speed_limit:.6f} "
                f"score={result.fisher_score:.6f} "
                f"clip={result.speed_clipped} "
                f"stop={result.should_stop}"
            )
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
        """Convenience wrapper for the simulation loop's authoritative c2w state."""
        return self.next_pose(
            gs_backend=gs_backend,
            current_viewpoint=np.asarray(current_c2w, dtype=np.float64),
            idx=idx,
            intrinsics_vec=intrinsics_vec,
            image_size=image_size,
        )
