"""Phase-2 bridge from simulated RGBD frames into OmniMap."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import torch

from .pose_utils import assert_pose_roundtrip, c2w_to_posevec


def _ensure_omnimap_import_paths() -> Path:
    """Ensure both repo root and `omnimap/` are importable.

    `omnimap/omni.py` currently imports sibling modules like `from util.utils ...`,
    so we need the `omnimap/` directory itself on `sys.path`.
    """
    repo_root = Path(__file__).resolve().parent.parent
    omnimap_dir = repo_root / "omnimap"
    for path in (repo_root, omnimap_dir):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
    return repo_root


def build_default_omnimap_args(
    output: str = "replica/output/sim_phase2",
    depth_scale: float = 1000.0,
    vis_gui: bool = False,
) -> SimpleNamespace:
    """Create the minimum args object required by `OMNI`."""
    return SimpleNamespace(
        output=output,
        depth_scale=float(depth_scale),
        vis_gui=bool(vis_gui),
        image_size=None,
    )


def load_omnimap_config(config_path: str | os.PathLike[str]) -> Dict[str, Any]:
    """Load OmniMap yaml config with the project's existing helper."""
    _ensure_omnimap_import_paths()
    from util.utils import load_config

    config = load_config(str(config_path))
    if not isinstance(config, dict):
        raise TypeError(f"Expected config dict, got {type(config)!r}")
    return config


def apply_config_overrides(
    config: Mapping[str, Any],
    overrides: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Apply a shallow set of top-level config overrides for sim-side debugging."""
    merged = dict(config)
    if not overrides:
        return merged
    for key, value in overrides.items():
        if value is not None:
            merged[key] = value
    return merged


def build_fisher_debug_config_overrides(
    *,
    show_fisher_heatmap: bool | None = None,
    show_fisher_arrows: bool | None = None,
    fisher_arrow_length: float | None = None,
    fisher_debug_log: bool | None = None,
    fisher_window_mode: str | None = None,
    fisher_heatmap_window_name: str | None = None,
    fisher_velocity_window_name: str | None = None,
    fisher_num_samples: int | None = None,
    fisher_num_dense_points: int | None = None,
    fisher_idw_power: float | None = None,
    fisher_display_radius_scale: float | None = None,
    fisher_arrow_radius_scale: float | None = None,
) -> Dict[str, Any]:
    """Translate sim-side Fisher debug toggles into OmniMap config overrides."""
    overrides: Dict[str, Any] = {}
    if show_fisher_heatmap is not None:
        overrides["show_fisher_heatmap"] = bool(show_fisher_heatmap)
    if show_fisher_arrows is not None:
        overrides["enable_velocity_field"] = bool(show_fisher_arrows)
        overrides["show_velocity_field"] = bool(show_fisher_arrows)
    if fisher_arrow_length is not None:
        overrides["velocity_arrow_length"] = float(fisher_arrow_length)
    if fisher_debug_log is not None:
        overrides["velocity_debug_log"] = bool(fisher_debug_log)
    if fisher_window_mode is not None:
        overrides["fisher_window_mode"] = str(fisher_window_mode)
    if fisher_heatmap_window_name is not None:
        overrides["fisher_heatmap_window_name"] = str(fisher_heatmap_window_name)
    if fisher_velocity_window_name is not None:
        overrides["fisher_velocity_window_name"] = str(fisher_velocity_window_name)
    if fisher_num_samples is not None:
        overrides["fisher_num_samples"] = int(fisher_num_samples)
    if fisher_num_dense_points is not None:
        overrides["fisher_num_dense_points"] = int(fisher_num_dense_points)
    if fisher_idw_power is not None:
        overrides["fisher_idw_power"] = float(fisher_idw_power)
    if fisher_display_radius_scale is not None:
        overrides["fisher_display_radius_scale"] = float(fisher_display_radius_scale)
    if fisher_arrow_radius_scale is not None:
        overrides["fisher_arrow_radius_scale"] = float(fisher_arrow_radius_scale)
    return overrides


class _NullProgressBar:
    """Small tqdm-like stub for direct Python function calls."""

    def __init__(self) -> None:
        self.n = 0
        self.last_postfix: Dict[str, Any] = {}

    def set_postfix(self, data: Mapping[str, Any]) -> None:
        self.last_postfix = dict(data)

    def update(self, step: int = 1) -> None:
        self.n += int(step)

    def close(self) -> None:
        return None


@dataclass
class StepResult:
    """Lightweight per-frame summary exposed back to the simulation loop."""

    idx: int
    camera_position: list[float]
    depth_min_m: float
    depth_max_m: float
    gs_initialized: bool
    num_keyframes: int
    num_gaussians: int
    latest_keyframe_uid: Optional[int]
    viewpoint: Any


class OmniMapRunner:
    """Feed simulated RGBD + pose directly into `OMNI.track(...)`."""

    def __init__(
        self,
        args: Any,
        config: Mapping[str, Any],
        *,
        max_depth_m: Optional[float] = None,
        progress_bar: Optional[Any] = None,
        verbose: bool = True,
    ) -> None:
        _ensure_omnimap_import_paths()
        from omni import OMNI

        if not hasattr(args, "output") or not hasattr(args, "depth_scale") or not hasattr(args, "vis_gui"):
            raise AttributeError(
                "args must at least provide `output`, `depth_scale`, and `vis_gui`"
            )

        self.args = args
        self.config = dict(config)
        self.max_depth_m = None if max_depth_m is None else float(max_depth_m)
        self.progress_bar = progress_bar or _NullProgressBar()
        self.verbose = bool(verbose)

        if self.args.output != "None":
            Path(self.args.output).mkdir(parents=True, exist_ok=True)

        self.omni = OMNI(self.args, self.config)

    @classmethod
    def from_config_path(
        cls,
        *,
        config_path: str | os.PathLike[str],
        output: str,
        depth_scale: float = 1000.0,
        vis_gui: bool = False,
        scene: Optional[str] = None,
        max_depth_m: Optional[float] = None,
        config_overrides: Optional[Mapping[str, Any]] = None,
        verbose: bool = True,
    ) -> "OmniMapRunner":
        args = build_default_omnimap_args(
            output=output,
            depth_scale=depth_scale,
            vis_gui=vis_gui,
        )
        config = load_omnimap_config(config_path)
        base_overrides: Dict[str, Any] = {}
        if scene is not None:
            base_overrides["scene"] = scene
        if config_overrides:
            base_overrides.update(dict(config_overrides))
        config = apply_config_overrides(config, base_overrides)
        return cls(
            args=args,
            config=config,
            max_depth_m=max_depth_m,
            verbose=verbose,
        )

    @staticmethod
    def _prepare_rgb(rgb: np.ndarray) -> torch.Tensor:
        """Convert a numpy RGB frame into OmniMap's `[3, H, W]` tensor layout."""
        rgb = np.asarray(rgb)
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"rgb must have shape [H, W, 3], got {rgb.shape}")
        if rgb.dtype != np.uint8:
            raise ValueError(f"rgb must be uint8, got {rgb.dtype}")
        return torch.as_tensor(np.ascontiguousarray(rgb)).permute(2, 0, 1)

    def _prepare_depth(self, depth_m: np.ndarray) -> torch.Tensor:
        """Normalize a metric depth map into a clean tensor for tracking."""
        depth_m = np.asarray(depth_m, dtype=np.float32)
        if depth_m.ndim != 2:
            raise ValueError(f"depth_m must have shape [H, W], got {depth_m.shape}")
        if not np.isfinite(depth_m).all():
            depth_m = depth_m.copy()
            depth_m[~np.isfinite(depth_m)] = 0.0
        depth_m = depth_m.copy()
        depth_m[depth_m < 0.0] = 0.0
        if self.max_depth_m is not None:
            depth_m[depth_m > self.max_depth_m] = 0.0
        return torch.as_tensor(np.ascontiguousarray(depth_m))

    @staticmethod
    def _prepare_intrinsics(intrinsics_vec: Sequence[float]) -> torch.Tensor:
        """Validate and pack `[fx, fy, cx, cy]` into a tensor."""
        intrinsics = np.asarray(intrinsics_vec, dtype=np.float64).reshape(-1)
        if intrinsics.shape != (4,):
            raise ValueError(
                f"intrinsics_vec must have shape (4,), got {intrinsics.shape}"
            )
        if not np.isfinite(intrinsics).all():
            raise ValueError("intrinsics_vec contains NaN or Inf")
        if intrinsics[0] <= 0 or intrinsics[1] <= 0:
            raise ValueError(f"fx/fy must be positive, got {intrinsics[:2].tolist()}")
        return torch.as_tensor(intrinsics)

    def step(
        self,
        idx: int,
        rgb: np.ndarray,
        depth_m: np.ndarray,
        c2w: np.ndarray,
        intrinsics_vec: Sequence[float],
        *,
        is_last: bool = False,
        update_rate: int = 1,
        run_pose_check: bool = True,
    ) -> StepResult:
        """Push one simulated frame into OmniMap."""
        c2w = np.asarray(c2w, dtype=np.float64)
        if c2w.shape != (4, 4):
            raise ValueError(f"c2w must have shape (4, 4), got {c2w.shape}")
        if run_pose_check:
            assert_pose_roundtrip(c2w)

        # The simulation side stays authoritative in metric RGBD / `c2w`,
        # and this bridge performs the exact conversions OmniMap expects.
        image = self._prepare_rgb(rgb)
        depth = self._prepare_depth(depth_m)
        intrinsics = self._prepare_intrinsics(intrinsics_vec)
        posevec = torch.as_tensor(c2w_to_posevec(c2w).astype(np.float32))
        pose_44 = torch.as_tensor(c2w.astype(np.float64))

        self.args.image_size = [int(image.shape[1]), int(image.shape[2])]

        # `OMNI.track(...)` is the only downstream tracking entry used by the simulated pipeline.
        self.omni.track(
            int(idx),
            image[None],
            depth[None],
            posevec[None],
            self.progress_bar,
            intrinsics=intrinsics[None],
            is_last=bool(is_last),
            pose_44=pose_44[None],
            update_rate=int(update_rate),
        )

        valid = np.isfinite(depth_m) & (depth_m > 0)
        depth_min = float(depth_m[valid].min()) if np.any(valid) else float("nan")
        depth_max = float(depth_m[valid].max()) if np.any(valid) else float("nan")
        gs = self.omni.gs
        num_keyframes = len(gs.keyviewpoints)
        num_gaussians = len(gs.gaussians.get_xyz)
        latest_viewpoint = gs.keyviewpoints[-1] if num_keyframes > 0 else None
        latest_uid = int(latest_viewpoint.uid) if latest_viewpoint is not None else None

        result = StepResult(
            idx=int(idx),
            camera_position=c2w[:3, 3].tolist(),
            depth_min_m=depth_min,
            depth_max_m=depth_max,
            gs_initialized=bool(gs.initialized),
            num_keyframes=num_keyframes,
            num_gaussians=num_gaussians,
            latest_keyframe_uid=latest_uid,
            viewpoint=latest_viewpoint,
        )

        if self.verbose:
            print(
                f"[OmniMapRunner] idx={result.idx} cam_pos={result.camera_position} "
                f"depth_min={result.depth_min_m:.4f} depth_max={result.depth_max_m:.4f} "
                f"initialized={result.gs_initialized} keyframes={result.num_keyframes} "
                f"gaussians={result.num_gaussians}"
            )
        return result

    def terminate(self) -> None:
        """Finalize OmniMap outputs when the current run produced a non-empty map."""
        gs = self.omni.gs
        num_keyframes = len(gs.keyviewpoints)
        num_gaussians = len(gs.gaussians.get_xyz)

        if num_keyframes == 0 or num_gaussians == 0:
            print(
                "[OmniMapRunner] Skip omni.terminate(): "
                f"keyframes={num_keyframes}, gaussians={num_gaussians}. "
                "The Phase-2 smoke test ran, but the current inputs did not produce "
                "a non-empty map for offline evaluation/export."
            )
            if hasattr(self.progress_bar, "close"):
                self.progress_bar.close()
            return

        self.omni.terminate()
        if hasattr(self.progress_bar, "close"):
            self.progress_bar.close()
