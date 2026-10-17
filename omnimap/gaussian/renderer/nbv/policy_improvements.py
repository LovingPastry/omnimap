"""Switchable fixes for the Fisher motion policy, each one ablatable on its own.

The original policy follows the raw Fisher gradient of a criterion designed for
picking discrete views in free space, and uses it for continuous motion in a
constrained workspace. Three things that criterion never needed become
mandatory on a real arm, and each is one section here:

* ``feasibility`` -- the gradient knows nothing about which views the arm can
  reach, so it climbs toward the hemisphere rim. Outward components at the
  elevation band / workspace box are suppressed (tangent-cone projection).
* ``fisher_mask`` -- every gaussian in the map is summed, including table /
  background gaussians outside the object region and gaussians that cannot be
  seen from a camera below the support plane. Their information gain cannot be
  collected, but it still pulls the camera.
* ``velocity_mapping`` -- the gradient magnitude was used as a speed, so the
  speed scale was an un-physical, hand-tuned constant and the limiter did the
  real work. ``normalized`` keeps the gradient only as a direction and sets the
  speed from a nominal m/s and an adaptive, scale-free gate.
* ``stop`` -- explicit termination: information saturation, view-sphere
  coverage, budgets, and stall.

Every section defaults to *disabled*, so an empty / missing ``nbv_policy``
config reproduces the original behaviour bit for bit.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field, fields
from typing import Any, Optional, Sequence

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class FeasibilityConfig:
    enabled: bool = False
    # Reachable elevation band of the camera on the view sphere (deg).
    phi_min_deg: float = 15.0
    phi_max_deg: float = 80.0
    # Width of the soft band inside each limit over which the outward command
    # component is ramped down to zero (deg / m).
    margin_deg: float = 8.0
    # Optional reachable camera-position box [x_min, x_max, y_min, y_max, z_min, z_max]
    # in the world frame (arm workspace). None = no box constraint.
    workspace_bounds: Optional[list] = None
    workspace_margin_m: float = 0.03
    # Also hard-clamp the integrated pose into the band (cartesian sim path).
    hard_clamp: bool = True


@dataclass
class FisherMaskConfig:
    enabled: bool = False
    # Region of interest for gaussians that may contribute to the score,
    # [x_min, x_max, y_min, y_max, z_min, z_max]. "tsdf" = use tsdf.spatial_bounds
    # (resolved by the wrapper), None = no ROI mask.
    roi_bounds: Any = None
    roi_padding_m: float = 0.0
    # Support plane (table top) height. Gaussians below it are dropped and a
    # camera below it sees nothing above it. None = no plane.
    support_plane_z: Optional[float] = None
    # Soft width of the plane visibility gate (m).
    plane_softness_m: float = 0.01


@dataclass
class VelocityMappingConfig:
    # "legacy": speed = fisher_step_scale * log1p(|g|/N) (original).
    # "normalized": direction = g/|g|, speed = v_nominal * |g| / (|g| + kappa).
    mode: str = "legacy"
    v_nominal_mps: float = 0.03
    kappa_window: int = 20
    # Lower bound on kappa relative to the first observed |g|; guards the very
    # first ticks where the window holds a single sample.
    kappa_min: float = 1e-9


@dataclass
class StopConfig:
    enabled: bool = False
    min_steps: int = 20
    # A: information saturation. Stop when the windowed mean Fisher score falls
    # below score_ratio * (max windowed mean seen so far).
    saturation_enabled: bool = True
    score_window: int = 10
    score_ratio: float = 0.10
    # B: view-sphere coverage. Fraction of directions in the feasible band that
    # lie within coverage_radius_deg of some visited camera direction.
    coverage_enabled: bool = True
    coverage_threshold: float = 0.90
    coverage_radius_deg: float = 25.0
    coverage_num_dirs: int = 400
    coverage_phi_min_deg: float = 15.0
    coverage_phi_max_deg: float = 80.0
    # C: budgets. None / <=0 disables each.
    max_steps: Optional[int] = None
    max_path_m: Optional[float] = None
    max_time_s: Optional[float] = None
    # Budget clock: False = sum of the policy's dt per tick (deterministic, sim);
    # True = wall-clock since the first tick (online planner, whose dt is nominal).
    wall_clock: bool = False
    # Once a criterion fires the monitor stays stopped until reset(), so an online
    # planner ticking at 30 Hz keeps publishing should_stop instead of restarting.
    latch: bool = True
    # D: stall. Commanded tangential speed below stall_speed_mps for
    # stall_steps consecutive ticks.
    stall_enabled: bool = False
    stall_speed_mps: float = 1e-3
    stall_steps: int = 15


@dataclass
class NBVPolicyConfig:
    gradient_space: str = "angle"
    feasibility: FeasibilityConfig = field(default_factory=FeasibilityConfig)
    fisher_mask: FisherMaskConfig = field(default_factory=FisherMaskConfig)
    velocity_mapping: VelocityMappingConfig = field(default_factory=VelocityMappingConfig)
    stop: StopConfig = field(default_factory=StopConfig)

    _SECTIONS = {
        "feasibility": FeasibilityConfig,
        "fisher_mask": FisherMaskConfig,
        "velocity_mapping": VelocityMappingConfig,
        "stop": StopConfig,
    }

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> "NBVPolicyConfig":
        data = dict(data or {})
        cfg = cls()
        unknown = []
        for key, value in data.items():
            if key == "gradient_space":
                cfg.gradient_space = str(value)
            elif key in cls._SECTIONS:
                section_cls = cls._SECTIONS[key]
                section = getattr(cfg, key)
                valid = {f.name for f in fields(section_cls)}
                for sub_key, sub_value in dict(value or {}).items():
                    if sub_key not in valid:
                        unknown.append(f"{key}.{sub_key}")
                        continue
                    setattr(section, sub_key, sub_value)
            else:
                unknown.append(key)
        if unknown:
            raise ValueError(f"unknown nbv_policy keys: {sorted(unknown)}")
        if cfg.gradient_space not in {"angle", "tangent"}:
            raise ValueError(f"gradient_space must be angle|tangent, got {cfg.gradient_space!r}")
        if cfg.velocity_mapping.mode not in {"legacy", "normalized"}:
            raise ValueError(
                f"velocity_mapping.mode must be legacy|normalized, got {cfg.velocity_mapping.mode!r}"
            )
        return cfg

    def to_dict(self) -> dict:
        out: dict[str, Any] = {"gradient_space": self.gradient_space}
        for key in self._SECTIONS:
            section = getattr(self, key)
            out[key] = {f.name: getattr(section, f.name) for f in fields(section)}
        return out

    @property
    def is_legacy(self) -> bool:
        return (
            self.gradient_space == "angle"
            and not self.feasibility.enabled
            and not self.fisher_mask.enabled
            and self.velocity_mapping.mode == "legacy"
            and not self.stop.enabled
        )


def set_nested(data: dict, dotted_key: str, value: Any) -> None:
    """Set ``a.b=value`` into a nested dict (used by CLI overrides)."""
    parts = dotted_key.split(".")
    cursor = data
    for part in parts[:-1]:
        cursor = cursor.setdefault(part, {})
    cursor[parts[-1]] = value


def parse_override_value(text: str) -> Any:
    import yaml

    return yaml.safe_load(text)


# ---------------------------------------------------------------------------
# Fix 1: feasibility (reachability tangent-cone projection)
# ---------------------------------------------------------------------------


def _bounds6(bounds: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    b = [float(v) for v in bounds]
    if len(b) != 6:
        raise ValueError(f"bounds must have 6 entries, got {bounds}")
    return np.array([b[0], b[2], b[4]]), np.array([b[1], b[3], b[5]])


class FeasibilityProjector:
    """Remove the part of a commanded tangential velocity that leaves the feasible set.

    Constraints are (outward unit normal, activation in [0, 1]); activation is 0
    deep inside the set and 1 at the limit. For each, the outward component of
    the world velocity is removed in proportion to the activation:
    ``v <- v - s * max(0, v.n) * n``. Inward motion is never touched, so the
    camera is free to move away from a limit -- it just cannot be driven past it.
    """

    def __init__(self, cfg: FeasibilityConfig) -> None:
        self.cfg = cfg
        self.phi_lo = math.radians(float(cfg.phi_min_deg))
        self.phi_hi = math.radians(float(cfg.phi_max_deg))
        self.margin = max(math.radians(float(cfg.margin_deg)), 1e-9)
        if not (0.0 <= self.phi_lo < self.phi_hi <= math.pi / 2.0):
            raise ValueError(
                f"invalid feasibility band [{cfg.phi_min_deg}, {cfg.phi_max_deg}] deg"
            )
        self.box = (
            _bounds6(cfg.workspace_bounds) if cfg.workspace_bounds is not None else None
        )
        self.box_margin = max(float(cfg.workspace_margin_m), 1e-9)

    def active_constraints(
        self,
        *,
        phi: float,
        north_hat: np.ndarray,
        position: np.ndarray,
    ) -> list[tuple[np.ndarray, float, str]]:
        out: list[tuple[np.ndarray, float, str]] = []
        north_hat = np.asarray(north_hat, dtype=np.float64).reshape(3)
        # Lower elevation limit: outward = -north.
        s_lo = float(np.clip(1.0 - (phi - self.phi_lo) / self.margin, 0.0, 1.0))
        if s_lo > 0.0:
            out.append((-north_hat, s_lo, "phi_min"))
        s_hi = float(np.clip(1.0 - (self.phi_hi - phi) / self.margin, 0.0, 1.0))
        if s_hi > 0.0:
            out.append((north_hat, s_hi, "phi_max"))
        if self.box is not None:
            lo, hi = self.box
            p = np.asarray(position, dtype=np.float64).reshape(3)
            for axis in range(3):
                for sign, gap in ((-1.0, p[axis] - lo[axis]), (1.0, hi[axis] - p[axis])):
                    s = float(np.clip(1.0 - gap / self.box_margin, 0.0, 1.0))
                    if s > 0.0:
                        normal = np.zeros(3)
                        normal[axis] = sign
                        out.append((normal, s, f"box_{'xyz'[axis]}{'-' if sign < 0 else '+'}"))
        return out

    def project_rates(
        self,
        *,
        rates: np.ndarray,
        basis_1: np.ndarray,
        basis_2: np.ndarray,
        radius: float,
        phi: float,
        north_hat: np.ndarray,
        position: np.ndarray,
    ) -> tuple[np.ndarray, dict]:
        """Project 2-D rates expressed on (basis_1, basis_2) with v = R*(u1 b1 + u2 b2)."""
        rates = np.asarray(rates, dtype=np.float64).reshape(2)
        b = np.stack(
            [np.asarray(basis_1, dtype=np.float64), np.asarray(basis_2, dtype=np.float64)],
            axis=1,
        ) * float(radius)
        v = b @ rates
        v_in = v.copy()
        constraints = self.active_constraints(phi=phi, north_hat=north_hat, position=position)
        names = []
        for normal, s, name in constraints:
            outward = float(v @ normal)
            if outward > 0.0:
                v = v - s * outward * normal
                names.append(name)
        if not names:
            return rates, {"active": "", "removed_ratio": 0.0}
        new_rates, *_ = np.linalg.lstsq(b, v, rcond=None)
        removed = float(np.linalg.norm(v_in - v) / max(np.linalg.norm(v_in), 1e-12))
        return new_rates.astype(np.float64), {
            "active": "+".join(names),
            "removed_ratio": removed,
        }

    def clamp_phi(self, phi: float) -> float:
        return float(np.clip(phi, self.phi_lo, self.phi_hi))


# ---------------------------------------------------------------------------
# Fix 2: Fisher ROI / visibility mask
# ---------------------------------------------------------------------------


class FisherContributionMask:
    """Per-gaussian weights on the Fisher contribution, plus a per-view gate.

    The contribution vector is laid out like ``cal_cur_hessian``'s flat output:
    ``[xyz (N*3, row-major), opacity (N)]``.
    """

    def __init__(self, cfg: FisherMaskConfig) -> None:
        self.cfg = cfg
        self.roi = None
        if cfg.roi_bounds is not None and not isinstance(cfg.roi_bounds, str):
            lo, hi = _bounds6(cfg.roi_bounds)
            pad = float(cfg.roi_padding_m)
            self.roi = (lo - pad, hi + pad)
        self.plane_z = None if cfg.support_plane_z is None else float(cfg.support_plane_z)
        self.softness = max(float(cfg.plane_softness_m), 1e-6)

    def gaussian_weights(self, xyz: torch.Tensor) -> torch.Tensor:
        xyz = xyz.detach().reshape(-1, 3)
        keep = torch.ones(xyz.shape[0], dtype=torch.bool, device=xyz.device)
        if self.roi is not None:
            lo = torch.as_tensor(self.roi[0], dtype=xyz.dtype, device=xyz.device)
            hi = torch.as_tensor(self.roi[1], dtype=xyz.dtype, device=xyz.device)
            keep &= ((xyz >= lo) & (xyz <= hi)).all(dim=1)
        if self.plane_z is not None:
            keep &= xyz[:, 2] >= self.plane_z
        return keep.to(xyz.dtype)

    @staticmethod
    def expand_to_contribution(weights: torch.Tensor) -> torch.Tensor:
        return torch.cat([weights.repeat_interleave(3), weights], dim=0)

    def view_gate(self, camera_center: torch.Tensor | np.ndarray) -> float:
        """Visibility of the (above-plane) object from a camera: 0 below the support plane."""
        if self.plane_z is None:
            return 1.0
        z = float(np.asarray(
            camera_center.detach().cpu().numpy() if isinstance(camera_center, torch.Tensor)
            else camera_center,
            dtype=np.float64,
        ).reshape(-1)[2])
        x = (z - self.plane_z) / self.softness
        return float(1.0 / (1.0 + math.exp(-max(min(x, 60.0), -60.0))))


# ---------------------------------------------------------------------------
# Fix 3: scale-free direction / rate separation
# ---------------------------------------------------------------------------


class AdaptiveRateMapper:
    """direction = g/|g| ; speed = v_nominal * |g| / (|g| + kappa), kappa = running median."""

    def __init__(self, cfg: VelocityMappingConfig) -> None:
        self.cfg = cfg
        self.history: deque[float] = deque(maxlen=max(int(cfg.kappa_window), 1))
        self.last_kappa = float("nan")
        self.last_gate = float("nan")

    def reset(self) -> None:
        self.history.clear()

    def map(self, grad: np.ndarray, radius: float) -> tuple[np.ndarray, dict]:
        grad = np.asarray(grad, dtype=np.float64).reshape(2)
        g = float(np.linalg.norm(grad))
        if not math.isfinite(g):
            g = 0.0
        self.history.append(g)
        kappa = max(float(np.median(np.asarray(self.history))), float(self.cfg.kappa_min))
        gate = g / (g + kappa) if g > 0.0 else 0.0
        speed = float(self.cfg.v_nominal_mps) * gate
        self.last_kappa, self.last_gate = kappa, gate
        if g <= 1e-15:
            return np.zeros(2), {"kappa": kappa, "gate": 0.0, "speed_mps": 0.0}
        # Rates are on the (basis_1, basis_2) chart with v = R*(u1 b1 + u2 b2);
        # dividing by R turns m/s into the rad/s the downstream chain expects.
        rates = (speed / max(float(radius), 1e-9)) * grad / g
        return rates, {"kappa": kappa, "gate": gate, "speed_mps": speed}


# ---------------------------------------------------------------------------
# Fix 4: stop criteria
# ---------------------------------------------------------------------------


def _fibonacci_band_dirs(n: int, phi_min: float, phi_max: float) -> np.ndarray:
    """Roughly uniform directions on the sphere restricted to an elevation band."""
    z_lo, z_hi = math.sin(phi_min), math.sin(phi_max)
    i = np.arange(n, dtype=np.float64) + 0.5
    z = z_lo + (z_hi - z_lo) * i / n
    golden = math.pi * (3.0 - math.sqrt(5.0))
    theta = golden * i
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, 1.0))
    return np.stack([r * np.cos(theta), r * np.sin(theta), z], axis=1)


class StopMonitor:
    """Stateful termination: saturation OR coverage OR budget OR stall."""

    def __init__(self, cfg: StopConfig) -> None:
        self.cfg = cfg
        self.reset()
        self.dirs = _fibonacci_band_dirs(
            int(cfg.coverage_num_dirs),
            math.radians(float(cfg.coverage_phi_min_deg)),
            math.radians(float(cfg.coverage_phi_max_deg)),
        )
        self.cos_radius = math.cos(math.radians(float(cfg.coverage_radius_deg)))

    def reset(self) -> None:
        self.steps = 0
        self.scores: deque[float] = deque(maxlen=max(int(self.cfg.score_window), 1))
        self.max_window_mean = -float("inf")
        self.path_m = 0.0
        self.elapsed_s = 0.0
        self.last_position: Optional[np.ndarray] = None
        self.covered: Optional[np.ndarray] = None
        self.stall_count = 0
        self.last_state: dict[str, float] = {}
        self.stopped_reason = ""
        self._t0: Optional[float] = None

    def update(
        self,
        *,
        score: float,
        position: np.ndarray,
        view_dir: np.ndarray,
        dt: float,
        commanded_speed_mps: float,
    ) -> tuple[bool, str, dict]:
        cfg = self.cfg
        self.steps += 1
        if cfg.wall_clock:
            import time

            now = time.monotonic()
            if self._t0 is None:
                self._t0 = now
            self.elapsed_s = now - self._t0
        else:
            self.elapsed_s += float(dt)
        position = np.asarray(position, dtype=np.float64).reshape(3)
        if self.last_position is not None:
            self.path_m += float(np.linalg.norm(position - self.last_position))
        self.last_position = position.copy()

        if math.isfinite(float(score)):
            self.scores.append(float(score))
        window_mean = float(np.mean(self.scores)) if self.scores else float("nan")
        if len(self.scores) == self.scores.maxlen and math.isfinite(window_mean):
            self.max_window_mean = max(self.max_window_mean, window_mean)
        saturation_ratio = (
            window_mean / self.max_window_mean
            if self.max_window_mean > 0 and math.isfinite(window_mean)
            else float("nan")
        )

        view_dir = np.asarray(view_dir, dtype=np.float64).reshape(3)
        view_dir = view_dir / max(np.linalg.norm(view_dir), 1e-12)
        hit = (self.dirs @ view_dir) >= self.cos_radius
        self.covered = hit if self.covered is None else (self.covered | hit)
        coverage = float(self.covered.mean())

        if float(commanded_speed_mps) < float(cfg.stall_speed_mps):
            self.stall_count += 1
        else:
            self.stall_count = 0

        state = {
            "stop_steps": float(self.steps),
            "stop_path_m": float(self.path_m),
            "stop_elapsed_s": float(self.elapsed_s),
            "stop_window_score": window_mean,
            "stop_saturation_ratio": float(saturation_ratio),
            "stop_view_coverage": coverage,
            "stop_stall_count": float(self.stall_count),
        }
        self.last_state = state
        if not cfg.enabled:
            return False, "", state
        if cfg.latch and self.stopped_reason:
            return True, self.stopped_reason, state
        stop, reason = self._check(coverage, saturation_ratio)
        if stop and cfg.latch:
            self.stopped_reason = reason
        return stop, reason, state

    def _check(self, coverage: float, saturation_ratio: float) -> tuple[bool, str]:
        cfg = self.cfg

        if cfg.max_steps is not None and int(cfg.max_steps) > 0 and self.steps >= int(cfg.max_steps):
            return True, "budget_steps"
        if cfg.max_path_m is not None and float(cfg.max_path_m) > 0 and self.path_m >= float(cfg.max_path_m):
            return True, "budget_path"
        if cfg.max_time_s is not None and float(cfg.max_time_s) > 0 and self.elapsed_s >= float(cfg.max_time_s):
            return True, "budget_time"
        if self.steps < int(cfg.min_steps):
            return False, ""
        if cfg.coverage_enabled and coverage >= float(cfg.coverage_threshold):
            return True, "view_coverage"
        if (
            cfg.saturation_enabled
            and math.isfinite(saturation_ratio)
            and saturation_ratio < float(cfg.score_ratio)
        ):
            return True, "info_saturated"
        if cfg.stall_enabled and self.stall_count >= int(cfg.stall_steps):
            return True, "stalled"
        return False, ""


# ---------------------------------------------------------------------------
# Wrapper helpers
# ---------------------------------------------------------------------------


def resolve_nbv_policy_dict(
    raw: Optional[dict],
    *,
    tsdf_bounds: Optional[Sequence[float]] = None,
    object_bounds: Optional[Sequence[float]] = None,
) -> dict:
    """Resolve symbolic placeholders in a raw ``nbv_policy`` dict.

    ``fisher_mask.roi_bounds``: ``"tsdf"`` -> ``tsdf.spatial_bounds``;
    ``"object"`` -> the simulator's object AABB (sim only).
    ``fisher_mask.support_plane_z``: ``"tsdf_min_z"`` / ``"object_min_z"`` -> the
    z_min of those boxes (the object rests on the support plane).
    """
    import copy

    data = copy.deepcopy(dict(raw or {}))
    mask = data.get("fisher_mask")
    if not isinstance(mask, dict):
        return data
    sources = {"tsdf": tsdf_bounds, "object": object_bounds}

    enabled = bool(mask.get("enabled", False))

    def _source(name: str):
        bounds = sources.get(name)
        if bounds is None:
            if not enabled:
                # A disabled mask must not make a config unloadable where the
                # referenced bounds do not exist (e.g. sim without spatial_bounds).
                return None
            raise ValueError(
                f"nbv_policy.fisher_mask refers to '{name}' bounds, which are not available here"
            )
        return [float(v) for v in bounds]

    roi = mask.get("roi_bounds")
    if isinstance(roi, str):
        mask["roi_bounds"] = _source(roi)
    plane = mask.get("support_plane_z")
    if isinstance(plane, str):
        if not plane.endswith("_min_z"):
            raise ValueError(f"unknown support_plane_z placeholder {plane!r}")
        bounds = _source(plane[: -len("_min_z")])
        mask["support_plane_z"] = None if bounds is None else bounds[4]
    return data


def nbv_policy_from_config(config: Optional[dict]) -> dict:
    """``config['nbv_policy']`` with ``tsdf`` placeholders resolved (online wrappers)."""
    config = config or {}
    tsdf_bounds = (config.get("tsdf") or {}).get("spatial_bounds")
    return resolve_nbv_policy_dict(config.get("nbv_policy"), tsdf_bounds=tsdf_bounds)
