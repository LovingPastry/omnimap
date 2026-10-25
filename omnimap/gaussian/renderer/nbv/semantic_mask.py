"""Associate the current Gaussian map with the mapper's existing semantics.

This module has no detector/ROS dependencies. Tracking creates a small tensor
bundle after updating geometry; Planning consumes that bundle without loading
YOLO, TAP, SBERT, or a live TSDF map.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


def associate_instance_ids(
    points: torch.Tensor,
    voxels: torch.Tensor,
    voxel_ids: torch.Tensor,
    max_distance: float,
) -> torch.Tensor:
    """Batched nearest-voxel association; 0 means unknown/unreliable.

    Keep unknown voxels in the search: removing them would let nearby labelled
    surfaces claim their Gaussians. Never transfer a label beyond max_distance.
    """
    from scipy.spatial import cKDTree

    if not math.isfinite(max_distance) or max_distance <= 0:
        raise ValueError("semantic association distance must be finite and positive")
    points = points.detach().reshape(-1, 3)
    voxels = voxels.detach().reshape(-1, 3)
    voxel_ids = voxel_ids.detach().reshape(-1)
    if len(voxels) != len(voxel_ids):
        raise ValueError("semantic voxel coordinates and IDs must have equal lengths")
    ids = torch.zeros(len(points), device=points.device, dtype=torch.long)
    if not len(points) or not len(voxels):
        return ids
    xyz = points.cpu().numpy()
    source = voxels.cpu().numpy()
    source_valid = np.isfinite(source).all(axis=1)
    query_valid = np.isfinite(xyz).all(axis=1)
    if not source_valid.any() or not query_valid.any():
        return ids
    tree = cKDTree(source[source_valid])
    distances, indices = tree.query(
        xyz[query_valid], k=1, distance_upper_bound=max_distance, workers=-1
    )
    matched = np.isfinite(distances) & (distances < max_distance)
    if matched.any():
        source_ids = voxel_ids.cpu().numpy()[source_valid]
        rows = np.flatnonzero(query_valid)[matched]
        labels = np.maximum(source_ids[indices[matched]], 0)
        ids[torch.as_tensor(rows, device=points.device)] = torch.as_tensor(
            labels, device=points.device, dtype=torch.long
        )
    return ids


def clone_gaussian_semantics(
    state: dict | None, *, num_gaussians: int, frame_idx: int | None = None
) -> dict | None:
    """Validate alignment and detach ownership at a snapshot boundary."""
    if state is None:
        return None
    ids, similarities = state["instance_ids"], state["similarities"]
    if ids.shape != (num_gaussians,) or similarities.shape != (num_gaussians,):
        raise ValueError("Gaussian semantic bundle is not aligned with the Gaussian map")
    if ids.dtype != torch.long or not torch.isfinite(similarities).all():
        raise ValueError("invalid Gaussian semantic IDs/similarities")
    if frame_idx is not None and int(state["frame_idx"]) != int(frame_idx):
        raise ValueError("Gaussian semantics and geometry come from different tracking frames")
    return {
        "query": str(state["query"]),
        "frame_idx": int(state["frame_idx"]),
        "instance_ids": ids.detach().clone(),
        "similarities": similarities.detach().clone(),
    }


def update_gaussian_semantics(backend: Any, *, frame_idx: int) -> None:
    """Refresh after *all* geometry updates, including densification/pruning.

    Only query embeddings are cached. Gaussian labels are recomputed every
    tracking frame, even when the number of Gaussians has not changed.
    """
    cfg = (backend.config.get("nbv_policy") or {}).get("fisher_mask") or {}
    query = cfg.get("roi_query", "cup")
    if not (cfg.get("enabled", False) and cfg.get("roi_query_mode") == "semantic"):
        backend.gaussian_semantics = None
        return
    if not isinstance(query, str) or not query.strip():
        raise ValueError("semantic Fisher masking requires a nonempty roi_query")
    xyz = backend.gaussians.get_xyz.detach()
    tsdfs = backend.tsdfs
    points, _, _, voxel_ids, _, _, _ = tsdfs.get_all_voxels(if_confidence=False)
    distance = cfg.get("semantic_max_distance_m")
    distance = float(tsdfs.fill_dis if distance is None else distance)
    ids = associate_instance_ids(xyz, points, voxel_ids, distance)
    features = tsdfs.instance_feature.detach()
    valid = (ids > 0) & (ids < len(features))
    ids[~valid] = 0
    similarities = torch.zeros(len(xyz), device=xyz.device, dtype=xyz.dtype)
    if valid.any():
        cache = getattr(tsdfs, "_fisher_query_embeddings", None)
        if cache is None:
            cache = tsdfs._fisher_query_embeddings = {}
        if query not in cache:
            cache[query] = tsdfs.sbert_model.encode(
                [query], convert_to_tensor=True, device=str(features.device),
                show_progress_bar=False,
            ).detach().reshape(-1)
        emb = F.normalize(cache[query].to(features), dim=0)
        # Fused features are weighted averages and need normalization again.
        per_instance = F.normalize(features, dim=-1) @ emb
        semantic_valid = torch.isfinite(per_instance) & (features.norm(dim=-1) > 0)
        valid &= semantic_valid.to(ids.device)[ids]
        ids[~valid] = 0
        similarities[valid] = per_instance.to(xyz)[ids[valid]].clamp(-1, 1)
    backend.gaussian_semantics = {
        "query": query,
        "frame_idx": int(frame_idx),
        "instance_ids": ids,
        "similarities": similarities,
    }
