from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Protocol

import torch

from gaussian.utils.camera_utils import Camera, HemisphereCamera


@dataclass
class FisherEvalResult:
    score: float
    stat: torch.Tensor
    raw_grad: Optional[torch.Tensor] = None
    debug_stats: Dict[str, object] = field(default_factory=dict)


@dataclass
class HemisphereFieldResult:
    idx: int
    base_hemi: HemisphereCamera
    history_stat: torch.Tensor
    sample_dirs: torch.Tensor
    sample_vals: torch.Tensor
    # Optional: per-sample 3D velocity directions (tangent to hemisphere surface).
    # Shape: [num_samples, 3].
    sample_vel_dirs: Optional[torch.Tensor] = None

    dense_dirs: torch.Tensor = None
    dense_vals: torch.Tensor = None
    dense_colors: torch.Tensor = None
    fisher_norm: torch.Tensor = None
    color_stats: Dict[str, float] = field(default_factory=dict)
    debug_stats: Dict[str, object] = field(default_factory=dict)


class FisherEvaluator(Protocol):
    def compute_history_stat(self, keyviewpoints) -> torch.Tensor:
        ...

    def compute_view_score(
        self, cam: Camera, history_stat: torch.Tensor
    ) -> FisherEvalResult:
        ...

    def compute_view_gradient(
        self, hemisphere_cam: HemisphereCamera, history_stat: torch.Tensor, eps: float = 0.01
    ) -> torch.Tensor:
        ...

    def build_hemisphere_field(
        self,
        viewpoint: Camera,
        scene_center: torch.Tensor,
        idx: int,
        num_samples: int,
        num_dense_points: int,
        power: float,
    ) -> HemisphereFieldResult:
        ...
