"""Semantic selection, geometry alignment, and frozen Planning transport.

Run with the InfoFlow Python. All tests except the real Gaussian snapshot
roundtrip run on CPU; that test additionally needs a working CUDA device.
"""

from __future__ import annotations

import copy
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "omnimap", ROOT / "info_flow"):
    sys.path.insert(0, str(path))

from omnimap.gaussian.renderer.nbv.semantic_mask import (
    associate_instance_ids, clone_gaussian_semantics, update_gaussian_semantics,
)
from omnimap.gaussian.renderer.nbv.policy_improvements import (
    FisherContributionMask, FisherMaskConfig, NBVPolicyConfig,
)
from omnimap.gaussian.renderer.nbv.legacy_fisher import LegacyFisherEvaluator


class TextEncoder:
    def __init__(self):
        self.calls = 0

    def encode(self, texts, **kwargs):
        self.calls += 1
        return torch.tensor([[1.0, 0.0]])


def backend_fixture():
    # A table Gaussian lies *inside* the cup's AABB. Semantic selection must
    # exclude it. IDs 1 and 3 are two distinct cup instances; both may match.
    points = torch.tensor([[-.1, 0, .3], [.1, 0, .3], [0, 0, .3], [.3, 0, .3]])
    ids = torch.tensor([1, 1, 2, 3])
    features = torch.tensor([[0., 0.], [.2, 0.], [0., 1.], [.3, .1]])
    tsdfs = NS(
        fill_dis=.02, instance_feature=features, sbert_model=TextEncoder(),
        get_all_voxels=lambda **kwargs: (points, None, None, ids, None, None, None),
    )
    xyz = torch.cat((points, torch.tensor([[3., 0., .3]])))
    cfg = {"nbv_policy": {"fisher_mask": {
        "enabled": True, "roi_query_mode": "semantic", "roi_query": "cup",
        "roi_query_min_similarity": .9,
    }}}
    return NS(config=cfg, gaussians=NS(get_xyz=xyz), tsdfs=tsdfs)


def mask_for(backend):
    cfg = NBVPolicyConfig.from_dict(backend.config["nbv_policy"]).fisher_mask
    return FisherContributionMask(cfg)


class SemanticTests(unittest.TestCase):
    def test_empty_config_defaults_to_dynamic_semantics(self):
        from omnimap.gaussian.renderer.nbv.policy_improvements import nbv_policy_from_config
        backend = backend_fixture()
        backend.config = {}
        update_gaussian_semantics(backend, frame_idx=11)
        cfg = NBVPolicyConfig.from_dict(nbv_policy_from_config({}))
        self.assertTrue(cfg.fisher_mask.enabled)
        self.assertEqual(cfg.fisher_mask.roi_query_mode, "semantic")
        mask = FisherContributionMask(cfg.fisher_mask)
        self.assertEqual(mask.gaussian_weights(backend.gaussians.get_xyz).sum(), 0)
        self.assertEqual(mask.gaussian_weights(backend.gaussians.get_xyz,
                         semantic_state=backend.gaussian_semantics).tolist(), [1, 1, 0, 1, 0])
        backend.config = {"nbv_policy": {"fisher_mask": {"enabled": False}}}
        update_gaussian_semantics(backend, frame_idx=12)
        self.assertIsNone(backend.gaussian_semantics)

    def test_real_voxel_export_preserves_unknown_and_unreliable_neighbors(self):
        # Execute the actual exporter method with a CPU-backed Open3D storage
        # adapter; no detector/GPU initialization or replacement filter logic.
        import ast
        tree = ast.parse((ROOT / "omnimap/tsdf_backend.py").read_text())
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "TSDFBackEnd")
        method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "get_all_voxels")
        namespace = {"torch": torch, "o3c": NS(cuda=NS(synchronize=lambda: None))}
        exec(compile(ast.Module(body=[method], type_ignores=[]), "tsdf_backend.py", "exec"), namespace)
        class TensorAdapter:
            def __init__(self, tensor): self.tensor = tensor
            def to_dlpack(self): return torch.utils.dlpack.to_dlpack(self.tensor)
        points = torch.zeros(8, 3)
        points[:, 0] = torch.arange(8) * .01
        indices = TensorAdapter(torch.tensor([0]))
        attributes = {"weight": torch.ones(1, 2, 2, 2, 1), "color": torch.zeros(1, 2, 2, 2, 3)}
        world = NS(hashmap=lambda: NS(active_buf_indices=lambda: indices),
                   voxel_coordinates_and_flattened_indices=lambda _: (TensorAdapter(points), None),
                   attribute=lambda name: TensorAdapter(attributes[name]))
        ids = torch.zeros(1, 2, 2, 2, 3, dtype=torch.long)
        ids.reshape(8, 3)[[0, 7], 0] = 1
        ids.reshape(8, 3)[1, 0] = 2  # instance observed too few times
        counts = torch.zeros(8)
        counts[[0, 1, 7]] = 5
        backend = NS(world=world, block_resolution=2, vox_min_count=3, ins_min_count=3,
                     ins_min_voxel=1, instance_count_vol=counts.reshape(1, 2, 2, 2),
                     instance_id_vol=ids, instance_pro_vol=torch.ones(1, 2, 2, 2, 4),
                     instance_fea_count=torch.tensor([[1], [5], [1]]), instance_colors=torch.zeros(3, 3))
        export = namespace["get_all_voxels"]
        old = export(backend)
        guarded = export(backend, include_unreliable=True)
        self.assertEqual(old[3].tolist(), [1, 1])
        self.assertEqual(guarded[3].tolist(), [1, 0, 0, 0, 0, 0, 0, 1])
        query = points[1:3]
        self.assertEqual(associate_instance_ids(query, old[0], old[3], .05).tolist(), [1, 1])
        self.assertEqual(associate_instance_ids(query, guarded[0], guarded[3], .05).tolist(), [0, 0])

    def test_batched_nearest_distance_unknown_and_nonfinite(self):
        points = torch.tensor([[0., 0, 0], [.1, 0, 0], [2., 0, 0], [float("nan"), 0, 0]])
        voxels = torch.tensor([[0., 0, 0], [.1, 0, 0], [float("nan"), 0, 0]])
        ids = associate_instance_ids(points, voxels, torch.tensor([1, 0, 2]), .02)
        self.assertEqual(ids.tolist(), [1, 0, 0, 0])
        self.assertEqual(associate_instance_ids(points, voxels[:0], torch.empty(0), .02).sum(), 0)

    def test_cup_membership_excludes_background_inside_bbox(self):
        backend = backend_fixture()
        update_gaussian_semantics(backend, frame_idx=7)
        mask = mask_for(backend)
        weights = mask.gaussian_weights(backend.gaussians.get_xyz, semantic_state=backend.gaussian_semantics)
        self.assertEqual(weights.tolist(), [1, 1, 0, 1, 0])
        self.assertEqual(mask.last_query_diag["semantic_target_ids"], [1, 3])
        # The fused feature norm is .2, but its cosine similarity is 1.
        self.assertAlmostEqual(backend.gaussian_semantics["similarities"][0].item(), 1)

    def test_reassociate_after_motion_pruning_insertion_and_reordering(self):
        backend = backend_fixture()
        update_gaussian_semantics(backend, frame_idx=0)
        backend.gaussians.get_xyz = backend.gaussians.get_xyz[[2, 0, 4]]
        update_gaussian_semantics(backend, frame_idx=1)
        self.assertEqual(backend.gaussian_semantics["instance_ids"].tolist(), [2, 1, 0])
        # Same length is insufficient for caching: centers can change in place.
        backend.gaussians.get_xyz[0] = torch.tensor([.3, 0, .3])
        update_gaussian_semantics(backend, frame_idx=2)
        self.assertEqual(backend.gaussian_semantics["instance_ids"].tolist(), [3, 1, 0])
        self.assertEqual(backend.tsdfs.sbert_model.calls, 1)

    def test_missing_target_does_not_fall_back_to_roi(self):
        backend = backend_fixture()
        mask = mask_for(backend)
        mask.roi = (np.full(3, -10), np.full(3, 10))
        self.assertEqual(mask.gaussian_weights(backend.gaussians.get_xyz).sum(), 0)
        update_gaussian_semantics(backend, frame_idx=1)
        backend.gaussian_semantics["similarities"].zero_()
        self.assertEqual(mask.gaussian_weights(backend.gaussians.get_xyz, semantic_state=backend.gaussian_semantics).sum(), 0)
        self.assertEqual(mask.last_query_diag["semantic_status"], "no_match")

    def test_bundle_rejects_wrong_geometry_frame_length_and_query(self):
        backend = backend_fixture()
        update_gaussian_semantics(backend, frame_idx=7)
        state = backend.gaussian_semantics
        for args in (dict(num_gaussians=4), dict(num_gaussians=5, frame_idx=8)):
            with self.assertRaises(ValueError):
                clone_gaussian_semantics(state, **args)
        cloned = clone_gaussian_semantics(state, num_gaussians=5, frame_idx=7)
        state["instance_ids"].zero_()
        self.assertEqual(cloned["instance_ids"][0], 1)
        cloned["query"] = "table"
        with self.assertRaises(ValueError):
            mask_for(backend).gaussian_weights(backend.gaussians.get_xyz, semantic_state=cloned)

    def test_background_fisher_cannot_change_selected_score_or_gradient(self):
        backend = backend_fixture()
        update_gaussian_semantics(backend, frame_idx=1)
        weights = mask_for(backend).gaussian_weights(
            backend.gaussians.get_xyz, semantic_state=backend.gaussian_semantics,
        )
        class Field(LegacyFisherEvaluator):
            background = 10.

            def _compute_current_raw_grad(self, cam):
                x = float(cam.theta)
                per_gaussian = torch.tensor([2 + x, 3 + x, self.background * (2 - x), 4 + x, self.background])
                return FisherContributionMask.expand_to_contribution(per_gaussian)

        evaluator = Field(None, {})
        history = torch.zeros(20)
        evaluator.set_score_mask(FisherContributionMask.expand_to_contribution(weights))
        def score(x):
            return evaluator.compute_view_score(NS(theta=x), history).score
        before = (score(0), (score(.01) - score(-.01)) / .02)
        evaluator.background = 1e8
        after = (score(0), (score(.01) - score(-.01)) / .02)
        np.testing.assert_allclose(before, after)
        self.assertAlmostEqual(before[0], 360)
        self.assertAlmostEqual(before[1], 120, delta=.01)

    def test_waiting_holds_all_motion_and_recovers_without_latching_stop(self):
        from test_nbv_policy_improvements import make_backend, POLICY_KW, c2w_at, INTR, SIZE
        from omnimap.gaussian.renderer.nbv.motion_policy import FisherMotionPolicy
        backend = make_backend([1, 0, .2], n_gauss=5)
        cfg = backend_fixture().config["nbv_policy"]
        cfg["stop"] = {"enabled": True, "stall_enabled": True, "stall_steps": 1}
        policy = FisherMotionPolicy(**POLICY_KW, nbv_config=cfg)
        current = c2w_at(0, .6)
        wait = policy.next_pose_from_c2w(gs_backend=backend, current_c2w=current, intrinsics_vec=INTR, image_size=SIZE, idx=1)
        self.assertEqual(wait.stop_reason, "semantic_target_unavailable")
        self.assertTrue(wait.should_stop)  # blocks Servo's radial/orientation loops too
        np.testing.assert_array_equal(wait.next_c2w, current)
        np.testing.assert_array_equal(wait.angular_velocity_world, [0, 0, 0])
        semantic = backend_fixture()
        update_gaussian_semantics(semantic, frame_idx=2)
        backend.gaussian_semantics = semantic.gaussian_semantics
        move = policy.next_pose_from_c2w(gs_backend=backend, current_c2w=current, intrinsics_vec=INTR, image_size=SIZE, idx=2)
        self.assertFalse(move.should_stop)
        self.assertGreater(np.linalg.norm(move.velocity_world), 0)

    def test_semantic_config_rejects_invalid_query_distance_threshold(self):
        for fields in ({"roi_query": None}, {"roi_query": " "},
                       {"semantic_max_distance_m": -1}, {"roi_query_min_similarity": float("nan")}):
            with self.assertRaises(ValueError):
                NBVPolicyConfig.from_dict({"fisher_mask": {"roi_query_mode": "semantic", **fields}})

    def test_semantic_mode_does_not_require_unused_tsdf_roi_bounds(self):
        from omnimap.gaussian.renderer.nbv.policy_improvements import nbv_policy_from_config
        config = backend_fixture().config
        config["nbv_policy"]["fisher_mask"]["roi_bounds"] = "tsdf"
        resolved = nbv_policy_from_config(config)  # no tsdf.spatial_bounds supplied
        self.assertIsNone(resolved["fisher_mask"]["roi_bounds"])


@unittest.skipUnless(torch.cuda.is_available(), "real Gaussian snapshot requires CUDA")
class SnapshotTests(unittest.TestCase):
    def test_real_gaussian_snapshot_roundtrip_is_frozen_and_semantic(self):
        from munch import munchify
        import yaml
        from planner_snapshot import build_planner_snapshot, serialize_planner_snapshot, deserialize_planner_snapshot, save_planner_snapshot_file, load_planner_snapshot_file
        from omnimap.gaussian.scene.gaussian_model import GaussianModel
        fixture = backend_fixture()
        cfg = yaml.safe_load((ROOT / "config/sim_rtabmap_config.yaml").read_text())
        cfg["nbv_policy"] = fixture.config["nbv_policy"]
        xyz = fixture.gaussians.get_xyz.cuda()
        gaussians = GaussianModel(0, cfg)
        gaussians.init_lr(6.)
        gaussians._xyz = torch.nn.Parameter(xyz.clone())
        gaussians._features_dc = torch.nn.Parameter(torch.zeros(len(xyz), 1, 3, device="cuda"))
        gaussians._features_rest = torch.nn.Parameter(torch.zeros(len(xyz), 0, 3, device="cuda"))
        gaussians._opacity = torch.nn.Parameter(torch.zeros(len(xyz), 1, device="cuda"))
        gaussians._rotation = torch.nn.Parameter(torch.tensor([[1., 0, 0, 0]], device="cuda").repeat(len(xyz), 1))
        gaussians._scaling = torch.nn.Parameter(torch.full((len(xyz), 3), -4., device="cuda"))
        gaussians.training_setup(munchify(cfg["opt_params"]))
        gaussians.max_radii2D = torch.zeros(len(xyz), device="cuda")
        live = NS(config=cfg, opt_params=munchify(cfg["opt_params"]), gaussians=gaussians,
                  tsdfs=fixture.tsdfs, fisher_eval=LegacyFisherEvaluator(gaussians, cfg), keyviewpoints=[],
                  get_runtime_device=lambda: "cuda:0")
        update_gaussian_semantics(live, frame_idx=7)
        snapshot = build_planner_snapshot(live_backend=live, model_version=1, keyframe_idx=7)
        live.gaussian_semantics["instance_ids"].zero_()
        self.assertEqual(snapshot.backend.gaussian_semantics["instance_ids"][0], 1)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "snapshot.pt"
            save_planner_snapshot_file(snapshot, path)
            loaded = load_planner_snapshot_file(path)
        weights = mask_for(fixture).gaussian_weights(loaded.backend.gaussians.get_xyz, semantic_state=loaded.backend.gaussian_semantics)
        self.assertEqual(weights.tolist(), [1, 1, 0, 1, 0])
        self.assertFalse(hasattr(loaded.backend, "tsdfs"))
        bundle = serialize_planner_snapshot(snapshot)
        broken = copy.copy(bundle)
        broken["keyframe_idx"] = 8
        with self.assertRaises(ValueError):
            deserialize_planner_snapshot(broken)
        for version in [1, 2, 3]:
            old = copy.copy(bundle)
            old["bundle_version"] = version
            old.pop("gaussian_semantics")
            self.assertIsNone(deserialize_planner_snapshot(old).backend.gaussian_semantics)


if __name__ == "__main__":
    unittest.main(verbosity=2)
