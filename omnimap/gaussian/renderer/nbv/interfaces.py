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
    dense_dirs: torch.Tensor
    dense_vals: torch.Tensor
    dense_colors: torch.Tensor
    fisher_norm: torch.Tensor
    color_stats: Dict[str, float]
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
