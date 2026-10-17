"""Checks for the switchable nbv_policy fixes. Runs without a GPU or pytest:

    python tests/test_nbv_policy_improvements.py

A fake backend scores views with an analytic field, so the whole control chain
of FisherMotionPolicy runs on CPU.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
for p in (REPO, REPO / "omnimap"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from omnimap.gaussian.renderer.nbv.motion_policy import FisherMotionPolicy  # noqa: E402
from omnimap.gaussian.renderer.nbv.policy_improvements import (  # noqa: E402
    AdaptiveRateMapper,
    FeasibilityConfig,
    FeasibilityProjector,
    FisherContributionMask,
    FisherMaskConfig,
    NBVPolicyConfig,
    StopConfig,
    StopMonitor,
    VelocityMappingConfig,
)
from omnimap.util.lookat import look_at_c2w  # noqa: E402

CENTER = np.zeros(3)
RADIUS = 0.6
INTR = [525.0, 525.0, 319.5, 239.5]
SIZE = (480, 640)
# Camera construction in camera_utils pins some tensors to CUDA when it exists.
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# fake backend: score = A * <n_hat, d_target> + B, gradients analytic
# ---------------------------------------------------------------------------


class FakeFisherEval:
    def __init__(self, target_dir, amplitude=1000.0):
        d = np.asarray(target_dir, dtype=np.float64)
        self.d = d / np.linalg.norm(d)
        self.amplitude = float(amplitude)
        self.precomputed_history_stat = torch.zeros(4)
        self.contribution_weight = None
        self.view_gate_fn = None
        self.mask_installs = 0

    def set_score_mask(self, w, gate):
        if w is not None:
            self.mask_installs += 1
        self.contribution_weight, self.view_gate_fn = w, gate

    def get_history_stat(self, keyviews, prefer_cached=True):
        return self.precomputed_history_stat

    def _field(self, theta, phi):
        n = np.array([math.cos(phi) * math.cos(theta), math.cos(phi) * math.sin(theta), math.sin(phi)])
        return self.amplitude * float(n @ self.d) + 2.0 * self.amplitude

    def compute_view_score(self, cam, history_stat):
        if hasattr(cam, "theta"):
            score = self._field(float(cam.theta.item()), float(cam.phi.item()))
        else:  # plain Camera (global saturation probe): angles from its centre
            c = cam.camera_center.detach().cpu().numpy().astype(np.float64)
            c = c / np.linalg.norm(c)
            score = self._field(math.atan2(c[1], c[0]), math.asin(np.clip(c[2], -1, 1)))
        if self.view_gate_fn is not None:
            score *= float(self.view_gate_fn(cam))
        return SimpleNamespace(score=score)

    def compute_view_gradient(self, cam, history_stat, eps=0.01):
        t, p = float(cam.theta.item()), float(cam.phi.item())
        dt = (self._field(t + eps, p) - self._field(t - eps, p)) / (2 * eps)
        dp = (self._field(t, p + eps) - self._field(t, p - eps)) / (2 * eps)
        return torch.tensor([dt, dp])

    def compute_view_gradient_tangent(self, cam, history_stat, eps=0.01):
        t, p = float(cam.theta.item()), float(cam.phi.item())
        n = np.array([math.cos(p) * math.cos(t), math.cos(p) * math.sin(t), math.sin(p)])
        east = np.array([-math.sin(t), math.cos(t), 0.0])
        north = np.array([-math.sin(p) * math.cos(t), -math.sin(p) * math.sin(t), math.cos(p)])
        g = self.amplitude * self.d
        g_t = g - (g @ n) * n
        return torch.tensor([float(g_t @ east), float(g_t @ north)])


def make_backend(target_dir, amplitude=1000.0, n_gauss=500):
    xyz = torch.randn(n_gauss, 3) * 0.1
    return SimpleNamespace(
        fisher_eval=FakeFisherEval(target_dir, amplitude),
        gaussians=SimpleNamespace(get_xyz=xyz),
        keyviewpoints=[],
        get_fisher_scene_center=lambda: torch.zeros(3, device=DEVICE),
        get_runtime_device=lambda: DEVICE,
    )


def c2w_at(theta, phi, radius=RADIUS):
    eye = CENTER + radius * np.array(
        [math.cos(phi) * math.cos(theta), math.cos(phi) * math.sin(theta), math.sin(phi)]
    )
    return look_at_c2w(eye=eye, target=CENTER)


def rollout(policy, backend, theta0, phi0, steps):
    c2w = c2w_at(theta0, phi0)
    out = []
    for i in range(steps):
        r = policy.next_pose_from_c2w(
            gs_backend=backend, current_c2w=c2w, intrinsics_vec=INTR, image_size=SIZE, idx=i
        )
        out.append(r)
        if r.should_stop:
            break
        c2w = r.next_c2w
    return out


POLICY_KW = dict(
    fisher_step_scale=0.03,
    cartesian=True,
    dt=0.1,
    radial_gain=2.0,
    linear_vel_max=0.5,
    spherical_speed_min=0.0,
    verbose=False,
)


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------


def test_config_defaults_are_legacy():
    assert NBVPolicyConfig.from_dict(None).is_legacy
    assert NBVPolicyConfig.from_dict({}).is_legacy
    cfg = NBVPolicyConfig.from_dict({"feasibility": {"enabled": True}})
    assert not cfg.is_legacy
    try:
        NBVPolicyConfig.from_dict({"feasibility": {"enabeld": True}})
    except ValueError:
        pass
    else:
        raise AssertionError("typo in key must be rejected")


def test_default_config_reproduces_legacy_chain():
    """nbv_config=None, {} and an explicit all-default dict give identical rollouts."""
    target = [1.0, 0.3, 0.2]
    a = rollout(FisherMotionPolicy(**POLICY_KW), make_backend(target), 0.2, 0.6, 15)
    b = rollout(
        FisherMotionPolicy(**POLICY_KW, nbv_config=NBVPolicyConfig().to_dict()),
        make_backend(target),
        0.2,
        0.6,
        15,
    )
    assert len(a) == len(b) == 15
    for ra, rb in zip(a, b):
        assert np.array_equal(ra.next_c2w, rb.next_c2w)
        assert ra.stop_reason == rb.stop_reason


def test_feasibility_removes_only_outward_component():
    proj = FeasibilityProjector(FeasibilityConfig(enabled=True, phi_min_deg=15, phi_max_deg=80, margin_deg=8))
    theta, phi = 0.0, math.radians(15.0)  # exactly at the lower limit
    east = np.array([-math.sin(theta), math.cos(theta), 0.0])
    north = np.array([-math.sin(phi) * math.cos(theta), -math.sin(phi) * math.sin(theta), math.cos(phi)])
    pos = RADIUS * np.array([math.cos(phi), 0.0, math.sin(phi)])
    # Downward + eastward command: downward part removed fully, east kept.
    rates, info = proj.project_rates(
        rates=np.array([0.1, -0.1]), basis_1=east, basis_2=north, radius=RADIUS,
        phi=phi, north_hat=north, position=pos,
    )
    assert info["active"] == "phi_min"
    assert abs(rates[1]) < 1e-12 and abs(rates[0] - 0.1) < 1e-12
    # Upward command is untouched.
    rates, info = proj.project_rates(
        rates=np.array([0.1, 0.1]), basis_1=east, basis_2=north, radius=RADIUS,
        phi=phi, north_hat=north, position=pos,
    )
    assert info["active"] == "" and np.allclose(rates, [0.1, 0.1])
    # Deep inside the band nothing is active.
    mid = math.radians(45)
    assert proj.active_constraints(phi=mid, north_hat=north, position=pos) == []


def test_feasibility_workspace_box():
    proj = FeasibilityProjector(
        FeasibilityConfig(enabled=True, workspace_bounds=[-1, 0.5, -1, 1, -1, 1], workspace_margin_m=0.05)
    )
    pos = np.array([0.5, 0.0, 0.3])  # on the x+ face
    e1, e2 = np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
    rates, info = proj.project_rates(
        rates=np.array([1.0, 1.0]), basis_1=e1, basis_2=e2, radius=1.0,
        phi=math.radians(45), north_hat=np.array([0.0, 0.0, 1.0]), position=pos,
    )
    assert "box_x+" in info["active"] and abs(rates[0]) < 1e-9 and abs(rates[1] - 1.0) < 1e-9


def test_mask_layout_and_gate():
    m = FisherContributionMask(FisherMaskConfig(enabled=True, roi_bounds=[-1, 1, -1, 1, 0, 1], support_plane_z=0.0))
    xyz = torch.tensor([[0.0, 0.0, 0.5], [2.0, 0.0, 0.5], [0.0, 0.0, -0.5]])
    w = m.gaussian_weights(xyz)
    assert w.tolist() == [1.0, 0.0, 0.0]
    full = FisherContributionMask.expand_to_contribution(w)
    assert full.tolist() == [1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0]
    assert m.view_gate(np.array([0, 0, 0.5])) > 0.99
    assert m.view_gate(np.array([0, 0, -0.5])) < 0.01


def test_rate_mapper_is_scale_invariant():
    cfg = VelocityMappingConfig(mode="normalized", v_nominal_mps=0.03, kappa_window=10)
    rng = np.random.default_rng(0)
    grads = rng.normal(size=(30, 2))
    a, b = AdaptiveRateMapper(cfg), AdaptiveRateMapper(cfg)
    for g in grads:
        ra, _ = a.map(g, 0.5)
        rb, _ = b.map(g * 1e4, 0.5)  # same field, 4 orders of magnitude larger
        assert np.allclose(ra, rb, rtol=1e-9, atol=1e-15)
        assert np.linalg.norm(ra) * 0.5 <= 0.03 + 1e-12  # never exceeds v_nominal


def test_stop_monitor_criteria():
    # coverage: visiting every direction closes coverage
    mon = StopMonitor(StopConfig(enabled=True, min_steps=1, saturation_enabled=False, coverage_threshold=0.9))
    stop = False
    for d in mon.dirs[::3]:
        stop, reason, _ = mon.update(score=1.0, position=d, view_dir=d, dt=0.1, commanded_speed_mps=0.01)
        if stop:
            break
    assert stop and reason == "view_coverage"
    # latched: a later tick still reports the same stop
    stop2, reason2, _ = mon.update(score=1.0, position=mon.dirs[0], view_dir=mon.dirs[0], dt=0.1, commanded_speed_mps=0.01)
    assert stop2 and reason2 == "view_coverage"
    # saturation: score decays to 5% of its windowed peak
    mon = StopMonitor(StopConfig(enabled=True, min_steps=5, coverage_enabled=False, score_window=3, score_ratio=0.1))
    reason = ""
    for k in range(60):
        stop, reason, _ = mon.update(
            score=1000.0 * 0.9 ** k, position=np.array([1.0, 0, 0]),
            view_dir=np.array([1.0, 0, 0]), dt=0.1, commanded_speed_mps=0.01,
        )
        if stop:
            break
    assert reason == "info_saturated" and 0.9 ** k < 0.2
    # local saturation is only a nomination: a reachable, still-informative view blocks it
    def decaying(probe_value):
        mon = StopMonitor(StopConfig(enabled=True, min_steps=5, coverage_enabled=False, score_window=3,
                                     score_ratio=0.1, global_check_every=1))
        for k in range(80):
            stop, reason, state = mon.update(
                score=1000.0 * 0.9 ** k, position=np.array([1.0, 0, 0]), view_dir=np.array([1.0, 0, 0]),
                dt=0.1, commanded_speed_mps=0.01, global_probe=lambda dirs: probe_value,
            )
            if stop:
                return reason, state
        return "", state
    reason, state = decaying(probe_value=500.0)
    assert reason == "" and state["stop_global_ratio"] > 0.1
    reason, state = decaying(probe_value=1.0)
    assert reason == "info_saturated" and state["stop_global_ratio"] < 0.1
    # a one-off collapse of every view (e.g. map opacity reset) is not confirmed
    mon = StopMonitor(StopConfig(enabled=True, min_steps=5, coverage_enabled=False, score_window=3,
                                 score_ratio=0.1, global_check_every=1, confirm_checks=2))
    for k in range(40):
        collapsed = k == 30
        stop, reason, _ = mon.update(
            score=1.0 if collapsed else 1000.0, position=np.array([1.0, 0, 0]), view_dir=np.array([1.0, 0, 0]),
            dt=0.1, commanded_speed_mps=0.01, global_probe=lambda dirs, c=collapsed: 1.0 if c else 900.0,
        )
        assert not stop
    # disabled monitor never stops but still reports state
    mon = StopMonitor(StopConfig(enabled=False, max_steps=1))
    stop, _, state = mon.update(score=1.0, position=np.ones(3), view_dir=np.ones(3), dt=0.1, commanded_speed_mps=0.0)
    assert not stop and "stop_view_coverage" in state


def test_feasibility_keeps_policy_off_the_rim():
    """Field pulls toward the equator; with feasibility the camera holds >= phi_min."""
    target = [1.0, 0.0, -0.8]  # below the horizon
    base = rollout(FisherMotionPolicy(**POLICY_KW), make_backend(target, 1e5), 0.0, 0.6, 80)
    fixed = rollout(
        FisherMotionPolicy(**POLICY_KW, nbv_config={"feasibility": {"enabled": True, "phi_min_deg": 15}}),
        make_backend(target),
        0.0,
        0.6,
        80,
    )
    base_min = min(r.next_phi for r in base)
    fixed_min = min(r.next_phi for r in fixed)
    assert base_min < math.radians(5), base_min
    assert fixed_min >= math.radians(15) - 1e-6, math.degrees(fixed_min)


def test_stop_zeroes_commands_and_mask_is_cleared():
    backend = make_backend([1.0, 0.0, 0.3])
    pol = FisherMotionPolicy(
        **POLICY_KW,
        nbv_config={
            "fisher_mask": {"enabled": True, "roi_bounds": [-1, 1, -1, 1, -1, 1]},
            "stop": {"enabled": True, "max_steps": 5},
        },
    )
    res = rollout(pol, backend, 0.0, 0.6, 20)
    assert len(res) == 5 and res[-1].should_stop and res[-1].stop_reason == "budget_steps"
    assert np.array_equal(res[-1].next_c2w, res[-1].current_c2w)
    assert np.linalg.norm(res[-1].velocity_world) == 0.0
    assert backend.fisher_eval.mask_installs == 5
    assert backend.fisher_eval.contribution_weight is None  # cleared after each tick
    assert "mask_kept_ratio" in res[0].nbv_diag


def test_normalized_speed_matches_nominal_scale():
    pol = FisherMotionPolicy(
        **POLICY_KW,
        gradient_space="tangent",
        nbv_config={"velocity_mapping": {"mode": "normalized", "v_nominal_mps": 0.04}},
    )
    res = rollout(pol, make_backend([0.2, 1.0, 0.4], amplitude=1e6), 0.0, 0.5, 10)
    speeds = [r.nbv_diag["tangent_speed_mps"] for r in res]
    assert max(speeds) <= 0.04 + 1e-9 and min(speeds) > 0.005, speeds


if __name__ == "__main__":
    failures = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(f"FAIL {name}: {type(exc).__name__}: {exc}")
    sys.exit(1 if failures else 0)
