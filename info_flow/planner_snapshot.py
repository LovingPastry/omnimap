from __future__ import annotations

import copy
import importlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from munch import munchify
from omnimap.util.utils import get_section_logger

try:
    from omnimap.gaussian.scene.gaussian_model import GaussianModel
except ImportError:
    from gaussian.scene.gaussian_model import GaussianModel


@dataclass
class PlannerSnapshot:
    model_version: int
    keyframe_idx: int
    created_wall_time: float
    backend: "FrozenPlannerBackend"


class FrozenPlannerBackend:
    """Read-only backend consumed by FisherMotionPolicy.

    It exposes the minimal interface FisherMotionPolicy needs while keeping
    model parameters detached from the online tracking backend.
    """

    def __init__(
        self,
        *,
        gaussians: GaussianModel,
        fisher_eval: Any,
        keyviewpoints: List[Any],
        scene_center: Optional[torch.Tensor],
        config: dict,
        runtime_device: str,
    ) -> None:
        self.gaussians = gaussians
        self.fisher_eval = fisher_eval
        self.keyviewpoints = keyviewpoints
        self.sence_center = scene_center
        self.config = config
        self._runtime_device = str(runtime_device)

    def get_fisher_scene_center(self) -> Optional[torch.Tensor]:
        center = self.sence_center
        if center is not None:
            return center

        xyz = getattr(self.gaussians, "get_xyz", None)
        if isinstance(xyz, torch.Tensor) and xyz.numel() > 0:
            xyz = xyz.reshape(-1, 3)
            finite_mask = torch.isfinite(xyz).all(dim=1)
            if torch.any(finite_mask):
                self.sence_center = xyz[finite_mask].mean(dim=0).detach().float()
                return self.sence_center

        if self.keyviewpoints:
            centers = []
            for viewpoint in self.keyviewpoints:
                camera_center = getattr(viewpoint, "camera_center", None)
                if camera_center is None:
                    continue
                camera_center = camera_center.detach().reshape(3)
                if torch.isfinite(camera_center).all():
                    centers.append(camera_center.float())
            if centers:
                self.sence_center = torch.stack(centers, dim=0).mean(dim=0)
                return self.sence_center

        return None

    def get_runtime_device(self) -> str:
        return self._runtime_device


SNAPSHOT_BUNDLE_VERSION = 1


def _clone_viewpoints(keyviewpoints: List[Any]) -> List[Any]:
    cloned = []
    for viewpoint in keyviewpoints:
        if hasattr(viewpoint, "clone") and callable(viewpoint.clone):
            cloned.append(viewpoint.clone())
        else:
            cloned.append(copy.deepcopy(viewpoint))
    return cloned


def _clone_gaussians(gaussians: GaussianModel, opt_params: Any) -> GaussianModel:
    cloned = GaussianModel(sh_degree=int(gaussians.max_sh_degree), config=gaussians.config)
    cloned.active_sh_degree = int(gaussians.active_sh_degree)
    cloned.init_lr(float(getattr(gaussians, "spatial_lr_scale", 1.0)))

    cloned._xyz = torch.nn.Parameter(gaussians._xyz.detach().clone())
    cloned._features_dc = torch.nn.Parameter(gaussians._features_dc.detach().clone())
    cloned._features_rest = torch.nn.Parameter(gaussians._features_rest.detach().clone())
    cloned._scaling = torch.nn.Parameter(gaussians._scaling.detach().clone())
    cloned._rotation = torch.nn.Parameter(gaussians._rotation.detach().clone())
    cloned._opacity = torch.nn.Parameter(gaussians._opacity.detach().clone())

    cloned.training_setup(opt_params)

    cloned.max_radii2D = gaussians.max_radii2D.detach().clone()
    cloned.xyz_gradient_accum = gaussians.xyz_gradient_accum.detach().clone()
    cloned.denom = gaussians.denom.detach().clone()
    if hasattr(gaussians, "n_obs") and isinstance(gaussians.n_obs, torch.Tensor):
        cloned.n_obs = gaussians.n_obs.detach().clone()
    if hasattr(gaussians, "instance_color") and isinstance(
        gaussians.instance_color, torch.Tensor
    ):
        cloned.instance_color = gaussians.instance_color.detach().clone()

    return cloned


def build_planner_snapshot(
    *,
    live_backend: Any,
    model_version: int,
    keyframe_idx: int,
) -> PlannerSnapshot:
    logger = get_section_logger("planner.snapshot", "planner")
    runtime_device = (
        live_backend.get_runtime_device()
        if hasattr(live_backend, "get_runtime_device")
        else ("cuda:0" if torch.cuda.is_available() else "cpu")
    )
    live_config = copy.deepcopy(dict(getattr(live_backend, "config", {})))
    live_opt_params = getattr(live_backend, "opt_params", None)
    if live_opt_params is None:
        live_opt_params = munchify(live_config.get("opt_params", {}))

    frozen_gaussians = _clone_gaussians(live_backend.gaussians, live_opt_params)
    fisher_eval_cls = type(live_backend.fisher_eval)
    frozen_fisher_eval = fisher_eval_cls(frozen_gaussians, live_config)
    frozen_keyviews = _clone_viewpoints(list(getattr(live_backend, "keyviewpoints", [])))

    scene_center = getattr(live_backend, "sence_center", None)
    if isinstance(scene_center, torch.Tensor):
        frozen_scene_center = scene_center.detach().clone().float()
    elif scene_center is None:
        frozen_scene_center = None
    else:
        frozen_scene_center = torch.as_tensor(np.asarray(scene_center), dtype=torch.float32)

    frozen_backend = FrozenPlannerBackend(
        gaussians=frozen_gaussians,
        fisher_eval=frozen_fisher_eval,
        keyviewpoints=frozen_keyviews,
        scene_center=frozen_scene_center,
        config=live_config,
        runtime_device=str(runtime_device),
    )

    snapshot = PlannerSnapshot(
        model_version=int(model_version),
        keyframe_idx=int(keyframe_idx),
        created_wall_time=float(time.monotonic()),
        backend=frozen_backend,
    )
    logger.debug(
        "已构建规划快照：model_version=%d keyframe_idx=%d keyviews=%d",
        int(snapshot.model_version),
        int(snapshot.keyframe_idx),
        len(frozen_keyviews),
    )
    return snapshot


def serialize_planner_snapshot(snapshot: PlannerSnapshot) -> Dict[str, Any]:
    backend = snapshot.backend
    fisher_eval_cls = type(backend.fisher_eval)
    scene_center = getattr(backend, "sence_center", None)
    if isinstance(scene_center, torch.Tensor):
        scene_center_bundle = scene_center.detach().clone().float()
    elif scene_center is None:
        scene_center_bundle = None
    else:
        scene_center_bundle = torch.as_tensor(
            np.asarray(scene_center),
            dtype=torch.float32,
        ).reshape(3)

    return {
        "bundle_version": int(SNAPSHOT_BUNDLE_VERSION),
        "model_version": int(snapshot.model_version),
        "keyframe_idx": int(snapshot.keyframe_idx),
        "created_wall_time": float(snapshot.created_wall_time),
        "config": copy.deepcopy(dict(getattr(backend, "config", {}))),
        "scene_center": scene_center_bundle,
        "runtime_device_hint": str(backend.get_runtime_device()),
        "fisher_eval_cls_module": str(fisher_eval_cls.__module__),
        "fisher_eval_cls_name": str(fisher_eval_cls.__name__),
        "gaussians": backend.gaussians,
        "keyviewpoints": list(getattr(backend, "keyviewpoints", [])),
    }


def deserialize_planner_snapshot(bundle: Dict[str, Any]) -> PlannerSnapshot:
    bundle_version = int(bundle.get("bundle_version", -1))
    if bundle_version != SNAPSHOT_BUNDLE_VERSION:
        raise ValueError(
            f"unsupported planner snapshot bundle version={bundle_version}, expected {SNAPSHOT_BUNDLE_VERSION}"
        )

    config = copy.deepcopy(dict(bundle.get("config", {})))
    gaussians = bundle["gaussians"]
    keyviewpoints = list(bundle.get("keyviewpoints", []))
    fisher_eval_module = importlib.import_module(bundle["fisher_eval_cls_module"])
    fisher_eval_cls = getattr(fisher_eval_module, bundle["fisher_eval_cls_name"])
    fisher_eval = fisher_eval_cls(gaussians, config)

    scene_center = bundle.get("scene_center", None)
    if isinstance(scene_center, torch.Tensor):
        frozen_scene_center = scene_center.detach().clone().float()
    elif scene_center is None:
        frozen_scene_center = None
    else:
        frozen_scene_center = torch.as_tensor(
            np.asarray(scene_center),
            dtype=torch.float32,
        ).reshape(3)

    backend = FrozenPlannerBackend(
        gaussians=gaussians,
        fisher_eval=fisher_eval,
        keyviewpoints=keyviewpoints,
        scene_center=frozen_scene_center,
        config=config,
        runtime_device=str(bundle.get("runtime_device_hint", "cpu")),
    )
    return PlannerSnapshot(
        model_version=int(bundle["model_version"]),
        keyframe_idx=int(bundle["keyframe_idx"]),
        created_wall_time=float(bundle["created_wall_time"]),
        backend=backend,
    )


def save_planner_snapshot_file(snapshot: PlannerSnapshot, path: str | Path) -> None:
    torch.save(serialize_planner_snapshot(snapshot), str(Path(path)))


def load_planner_snapshot_file(path: str | Path) -> PlannerSnapshot:
    bundle = torch.load(str(Path(path)), weights_only=False)
    if not isinstance(bundle, dict):
        raise TypeError(f"planner snapshot file must contain dict bundle, got {type(bundle)!r}")
    return deserialize_planner_snapshot(bundle)
