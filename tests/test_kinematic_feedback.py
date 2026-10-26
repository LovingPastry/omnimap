"""Numerical and feedback gates; no ROS, GPU, or robot required.

Run with: python tests/test_kinematic_feedback.py
"""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "info_flow"))
from kinematic_feedback import KinematicFeedbackGate, MOVEIT_STATUS_TEXT
from kinematic_projection import project_spherical_command


def feedback(stamp=100.0, **fields):
    values = dict(header=NS(stamp=NS(to_sec=lambda: stamp)), planner_tick=42,
                  source_frame="base_link", task_frame="tool0", sigma_min=0.1,
                  sigma_max=1.0, condition_number=10.0, manipulability=0.1,
                  command_speed_norm=0.02, actual_speed_norm=0.01,
                  transform_chain_valid=True, moveit_status=0,
                  moveit_status_text="", reason="")
    values.update(fields)
    return NS(**values)


class FeedbackTests(unittest.TestCase):
    def decision(self, msg, *, ros_now=100.0, wall_now=100.0):
        gate = KinematicFeedbackGate(deceleration_duration_sec=0.0)
        gate.update(msg, ros_now=100.0, wall_now=100.0)
        return gate.decision(ros_now=ros_now, wall_now=wall_now)

    def test_missing_and_independent_clock_timeouts(self):
        self.assertTrue(KinematicFeedbackGate().decision(ros_now=100, wall_now=100).should_stop)
        for ros_now, wall_now in ((101.0, 100.0), (100.0, 101.0)):
            with self.subTest(ros_now=ros_now, wall_now=wall_now):
                result = self.decision(feedback(), ros_now=ros_now, wall_now=wall_now)
                self.assertEqual(result.reason, "kinematic_status_stale")
        # Delayed messages cannot become fresh simply by arriving now.
        self.assertEqual(self.decision(feedback(stamp=90)).reason, "kinematic_status_stale")

    def test_bad_stamps_frames_chain_and_metrics_stop(self):
        cases = [
            (feedback(stamp=0), "invalid_stamp"),
            (feedback(stamp=101), "future"),
            (feedback(source_frame="camera"), "source_frame_mismatch"),
            (feedback(task_frame="camera"), "task_frame_mismatch"),
            (feedback(transform_chain_valid=False), "transform_chain_invalid"),
            (feedback(sigma_min=float("nan")), "metrics_invalid"),
            (feedback(sigma_min=-1), "metrics_invalid"),
            (feedback(sigma_min=2), "metrics_invalid"),
            (feedback(condition_number=0), "metrics_invalid"),
            (feedback(command_speed_norm=float("inf")), "metrics_invalid"),
        ]
        for msg, suffix in cases:
            with self.subTest(suffix=suffix, msg=msg):
                result = self.decision(msg)
                self.assertTrue(result.should_stop)
                self.assertTrue(result.reason.endswith(suffix))
        self.assertFalse(self.decision(feedback(source_frame="/base_link")).should_stop)

    def test_moveit_codes_and_singular_metrics(self):
        for status in (0, 1, 2, 3, 4, 5, -1, 99):
            with self.subTest(status=status):
                result = self.decision(feedback(moveit_status=status))
                self.assertEqual(result.should_stop, status not in (0, 1))
                if status not in (0, 1):
                    self.assertIn(MOVEIT_STATUS_TEXT.get(status, "UNKNOWN").lower(), result.reason)
        result = self.decision(feedback(sigma_min=0, condition_number=float("inf"), moveit_status=2))
        self.assertIn("halt_for_singularity", result.reason)

    def test_persistent_deceleration_recovery_and_gap(self):
        gate = KinematicFeedbackGate(deceleration_duration_sec=0.1)
        gate.update(feedback(moveit_status=1), ros_now=100, wall_now=100)
        self.assertEqual(gate.decision(ros_now=100.05, wall_now=100.05).speed_scale, 1)
        gate.update(feedback(stamp=100.2, moveit_status=1), ros_now=100.2, wall_now=100.2)
        self.assertEqual(gate.decision(ros_now=100.2, wall_now=100.2).speed_scale, 0.5)
        gate.update(feedback(stamp=100.3, moveit_status=0), ros_now=100.3, wall_now=100.3)
        self.assertEqual(gate.decision(ros_now=100.3, wall_now=100.3).speed_scale, 1)
        gate.update(feedback(stamp=101, moveit_status=1), ros_now=101, wall_now=101)
        self.assertEqual(gate.decision(ros_now=101, wall_now=101).speed_scale, 1)
        gate.update(feedback(stamp=102, moveit_status=1), ros_now=102, wall_now=102)
        self.assertEqual(gate.decision(ros_now=102, wall_now=102).speed_scale, 1)

    def test_invalid_configuration_rejected(self):
        for args in (dict(source_frame=""), dict(task_frame="/"), dict(timeout_sec=0),
                     dict(timeout_sec=float("nan")), dict(deceleration_duration_sec=-1),
                     dict(deceleration_scale=1.5)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                KinematicFeedbackGate(**args)


class ProjectionTests(unittest.TestCase):
    def test_feasible_is_unchanged(self):
        result, norm = project_spherical_command([1, 2], [1, 0], 0.5)
        np.testing.assert_array_equal(result, [1, 2])
        self.assertEqual(norm, 0)

    def test_projection_satisfies_constraint_and_is_closest(self):
        rng = np.random.default_rng(13)
        for _ in range(50):
            u, a = rng.normal(size=(2, 2))
            c = float(a @ u + 0.5)
            projected, norm = project_spherical_command(u, a, c)
            self.assertAlmostEqual(float(a @ projected), c)
            self.assertAlmostEqual(norm, np.linalg.norm(projected - u))
            # All other points on the boundary differ by a tangent component.
            tangent = np.array([-a[1], a[0]])
            for step in (-1, -0.1, 0.1, 1):
                self.assertGreater(np.linalg.norm(projected + step * tangent - u), norm)

    def test_bad_or_unusable_directions_raise(self):
        for u, a, c in (([0, 0], [0, 0], 1), ([0, 0], [1e-7, 0], 1),
                        ([0, 0, 0], [1, 0, 0], 1), ([0, 0], [[1, 0]], 1),
                        ([0, np.nan], [1, 0], 1), ([0, 0], [np.inf, 0], 1),
                        ([0, 0], [1, 0], np.nan), ([0, 0], [1, 0], [1])):
            with self.subTest(u=u, a=a, c=c), self.assertRaises(ValueError):
                project_spherical_command(u, a, c)


class RateModelTests(unittest.TestCase):
    def model(self, **overrides):
        from kinematic_projection import camera_rate_model
        fields = dict(radius=0.5, e_theta=[1, 0, 0], e_phi=[0, 1, 0],
                      rates=[0.04, -0.02], radial_velocity=[0, 0, 0.003],
                      linear_command=[0.02, -0.01, 0.003], angular_command=[0.1, -0.2, 0.3],
                      previous_linear=[0.019, -0.01, 0.003], linear_speed_limit=0.1,
                      linear_step_limit=0.02, linear_deadband=0.001)
        fields.update(overrides)
        return camera_rate_model(**fields)

    def test_affine_map_includes_radial_and_actual_angular_bias(self):
        valid, reason, matrix, bias = self.model()
        self.assertTrue(valid, reason)
        np.testing.assert_allclose(matrix @ [0.04, -0.02] + bias,
                                   [0.02, -0.01, 0.003, 0.1, -0.2, 0.3])
        # A different rate perturbation changes translation only at frozen pose.
        np.testing.assert_allclose(matrix @ [0.002, -0.004], [0.001, -0.002, 0, 0, 0, 0])

    def test_limit_branches_and_boundaries_are_not_affine_models(self):
        for fields in (dict(linear_speed_limit=0.01), dict(linear_step_limit=0.0001),
                       dict(linear_deadband=0.03), dict(linear_command=[0, 0, 0]),
                       dict(linear_speed_limit=np.linalg.norm([0.02, -0.01, 0.003])),
                       dict(radius=0), dict(rates=[float('nan'), 0])):
            with self.subTest(fields=fields):
                self.assertFalse(self.model(**fields)[0])

    def test_unclipped_preview_does_not_promote_invalid_execution_models(self):
        from kinematic_projection import unclipped_preview_rate_model
        source = self.model(linear_speed_limit=.01)
        self.assertFalse(source[0])
        preview = unclipped_preview_rate_model(source, [.1, -.05])
        self.assertTrue(preview[0])
        self.assertIn("unclipped_preview", preview[1])
        self.assertFalse(source[0])
        for invalid in (self.model(radius=0), self.model(linear_command=[0, 0, 0]),
                        (False, "unrecognized", source[2], source[3])):
            self.assertFalse(unclipped_preview_rate_model(invalid, [.1, -.05])[0])

    def test_disabled_limits_follow_runtime_semantics(self):
        self.assertTrue(self.model(linear_speed_limit=0, linear_step_limit=0, linear_deadband=0)[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
