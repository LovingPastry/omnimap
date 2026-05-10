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
        history_stat: Optional[torch.Tensor],
        scene_center: Optional[torch.Tensor],
        tsdf_geometry: Optional[Dict[str, Any]],
        config: dict,
        runtime_device: str,
    ) -> None:
        self.gaussians = gaussians
        self.fisher_eval = fisher_eval
        self.keyviewpoints = keyviewpoints
        self.history_stat = history_stat
        self.sence_center = scene_center
        self.tsdf_geometry = _normalize_tsdf_geometry(tsdf_geometry)
        self.config = config
        self._runtime_device = str(runtime_device)
        if hasattr(self.fisher_eval, "keyviewpoints"):
            self.fisher_eval.keyviewpoints = self.keyviewpoints
        if hasattr(self.fisher_eval, "set_precomputed_history_stat"):
            self.fisher_eval.set_precomputed_history_stat(self.history_stat)

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


SNAPSHOT_BUNDLE_VERSION = 3


def _tensor_vec3_or_none(value: Any) -> Optional[torch.Tensor]:
    if value is None:
        return None
    tensor = value.detach().clone().float() if isinstance(value, torch.Tensor) else torch.as_tensor(np.asarray(value), dtype=torch.float32)
    tensor = tensor.reshape(3)
    if not torch.isfinite(tensor).all():
        return None
    return tensor


def _normalize_tsdf_geometry(geometry: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(geometry, dict):
        return None
    center = _tensor_vec3_or_none(geometry.get("center"))
    bounds_min = _tensor_vec3_or_none(geometry.get("bounds_min"))
    bounds_max = _tensor_vec3_or_none(geometry.get("bounds_max"))
    if center is None or bounds_min is None or bounds_max is None:
        return None
    diag = geometry.get("diag", None)
    if diag is None:
        diag = torch.linalg.norm(bounds_max - bounds_min)
    diag = float(diag.detach().cpu().item()) if isinstance(diag, torch.Tensor) else float(diag)
    if not np.isfinite(diag) or diag <= 0.0:
        return None
    return {
        "center": center,
        "bounds_min": bounds_min,
        "bounds_max": bounds_max,
        "diag": diag,
        "num_points": int(geometry.get("num_points", 0) or 0),
    }


def _extract_live_tsdf_geometry(live_backend: Any) -> Optional[Dict[str, Any]]:
    tsdfs = getattr(live_backend, "tsdfs", None)
    if tsdfs is None:
        return None
    if hasattr(tsdfs, "get_pointcloud_geometry"):
        try:
            return _normalize_tsdf_geometry(tsdfs.get_pointcloud_geometry())
        except Exception:
            return None
    if not hasattr(tsdfs, "get_all_voxels"):
        return None
    try:
        points, _, _, _, _, _, _ = tsdfs.get_all_voxels(if_confidence=False)
    except Exception:
        return None
    if points is None or points.shape[0] == 0:
        return None
    finite_mask = torch.isfinite(points).all(dim=1)
    points = points[finite_mask]
    if points.shape[0] == 0:
        return None
    bounds_min = points.min(dim=0).values
    bounds_max = points.max(dim=0).values
    return _normalize_tsdf_geometry(
        {
            "center": points.mean(dim=0),
            "bounds_min": bounds_min,
            "bounds_max": bounds_max,
            "diag": torch.linalg.norm(bounds_max - bounds_min),
            "num_points": int(points.shape[0]),
        }
    )


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
    frozen_keyviews = _clone_viewpoints(list(getattr(live_backend, "keyviewpoints", [])))
    fisher_eval_cls = type(live_backend.fisher_eval)
    frozen_fisher_eval = fisher_eval_cls(frozen_gaussians, live_config)
    frozen_fisher_eval.keyviewpoints = frozen_keyviews
    frozen_history_stat = frozen_fisher_eval.compute_history_stat(frozen_keyviews)
    if hasattr(frozen_fisher_eval, "set_precomputed_history_stat"):
        frozen_fisher_eval.set_precomputed_history_stat(frozen_history_stat)

    scene_center = getattr(live_backend, "sence_center", None)
    if isinstance(scene_center, torch.Tensor):
        frozen_scene_center = scene_center.detach().clone().float()
    elif scene_center is None:
        frozen_scene_center = None
    else:
        frozen_scene_center = torch.as_tensor(np.asarray(scene_center), dtype=torch.float32)
    tsdf_geometry = _extract_live_tsdf_geometry(live_backend)

    frozen_backend = FrozenPlannerBackend(
        gaussians=frozen_gaussians,
        fisher_eval=frozen_fisher_eval,
        keyviewpoints=frozen_keyviews,
        history_stat=frozen_history_stat,
        scene_center=frozen_scene_center,
        tsdf_geometry=tsdf_geometry,
        config=live_config,
        runtime_device=str(runtime_device),
    )

    snapshot = PlannerSnapshot(
        model_version=int(model_version),
        keyframe_idx=int(keyframe_idx),
        created_wall_time=float(time.monotonic()),
        backend=frozen_backend,
    )
    if tsdf_geometry is not None:
        logger.info(
            "snapshot_tsdf_geometry: model_version=%d points=%d center=[%.3f %.3f %.3f] bounds_min=[%.3f %.3f %.3f] bounds_max=[%.3f %.3f %.3f] diag=%.3fm",
            int(snapshot.model_version),
            int(tsdf_geometry.get("num_points", 0)),
            float(tsdf_geometry["center"][0]),
            float(tsdf_geometry["center"][1]),
            float(tsdf_geometry["center"][2]),
            float(tsdf_geometry["bounds_min"][0]),
            float(tsdf_geometry["bounds_min"][1]),
            float(tsdf_geometry["bounds_min"][2]),
            float(tsdf_geometry["bounds_max"][0]),
            float(tsdf_geometry["bounds_max"][1]),
            float(tsdf_geometry["bounds_max"][2]),
            float(tsdf_geometry["diag"]),
        )
    else:
        logger.info(
            "snapshot_tsdf_geometry: model_version=%d missing",
            int(snapshot.model_version),
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
        "tsdf_geometry": _normalize_tsdf_geometry(
            getattr(backend, "tsdf_geometry", None)
        ),
        "runtime_device_hint": str(backend.get_runtime_device()),
        "fisher_eval_cls_module": str(fisher_eval_cls.__module__),
        "fisher_eval_cls_name": str(fisher_eval_cls.__name__),
        "gaussians": backend.gaussians,
        "keyviewpoints": list(getattr(backend, "keyviewpoints", [])),
        "history_stat": getattr(backend, "history_stat", None),
    }


def deserialize_planner_snapshot(bundle: Dict[str, Any]) -> PlannerSnapshot:
    bundle_version = int(bundle.get("bundle_version", -1))
    if bundle_version not in {1, 2, SNAPSHOT_BUNDLE_VERSION}:
        raise ValueError(
            f"unsupported planner snapshot bundle version={bundle_version}, expected 1, 2, or {SNAPSHOT_BUNDLE_VERSION}"
        )

    config = copy.deepcopy(dict(bundle.get("config", {})))
    gaussians = bundle["gaussians"]
    keyviewpoints = list(bundle.get("keyviewpoints", []))
    history_stat = bundle.get("history_stat", None) if bundle_version >= 2 else None
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
    tsdf_geometry = _normalize_tsdf_geometry(bundle.get("tsdf_geometry", None))

    backend = FrozenPlannerBackend(
        gaussians=gaussians,
        fisher_eval=fisher_eval,
        keyviewpoints=keyviewpoints,
        history_stat=history_stat,
        scene_center=frozen_scene_center,
        tsdf_geometry=tsdf_geometry,
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
