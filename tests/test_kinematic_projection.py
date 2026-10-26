"""Analytic convex optima and failures; no ROS or hardware dependency."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "info_flow"))
from kinematic_projection import project_feasible_spherical_command as project


class FeasibleProjectionTests(unittest.TestCase):
    def test_halfspace_analytic_optimum(self):
        point, distance = project([0, 0], [1, 1], 2)
        np.testing.assert_allclose(point, [1, 1])
        self.assertAlmostEqual(distance, np.sqrt(2))

    def test_joint_speed_limit_must_be_solved_with_barrier(self):
        # Naively clipping the half-space result (1,1) to x<=.5 loses x+y>=2.
        point, _ = project([0, 0], [1, 1], 2,
                           linear_matrix=[[1, 0], [-1, 0]], linear_bounds=[.5, .5])
        np.testing.assert_allclose(point, [.5, 1.5], atol=1e-9)

    def test_speed_and_acceleration_balls_joint_optimum(self):
        # Velocity speed <=1; per-step change from (1,0) <=1. Closest point
        # to (0,2) lies at the intersection of these two circles.
        point, _ = project([0, 2], [0, 1], 0,
                           norm_limits=[(np.eye(2), [0, 0], 1),
                                        (np.eye(2), [-1, 0], 1)])
        np.testing.assert_allclose(point, [.5, np.sqrt(3)/2], atol=1e-7)

    def test_local_ball_cannot_be_applied_after_projection(self):
        # The unconstrained half-space optimum is outside the local ball,
        # but a feasible correction exists on its boundary.
        point, _ = project([0, 1], [1, 0], .8, local_center=[0, 0], local_radius=1)
        np.testing.assert_allclose(point, [.8, .6], atol=1e-7)

    def test_affine_physical_model_and_input_ownership(self):
        original = np.array([2., 0.])
        matrix = np.array([[.1, 0], [0, .1], [0, 0]])
        point, _ = project(original, [1, 0], 0,
                           norm_limits=[(matrix, [.05, 0, 0], .1)])
        np.testing.assert_allclose(point, [.5, 0], atol=1e-7)
        np.testing.assert_array_equal(original, [2, 0])

    def test_infeasible_intersection_rejected(self):
        for extra in (dict(local_center=[0, 0], local_radius=.5),
                      dict(linear_matrix=[[1, 0]], linear_bounds=[.5]),
                      dict(norm_limits=[(np.eye(2), [0, 0], .5)])):
            with self.assertRaises(ValueError):
                project([0, 0], [1, 0], 1, **extra)

    def test_invalid_set_rejected(self):
        for extra in (dict(linear_matrix=[[1, 0]]),
                      dict(linear_matrix=[[1, np.nan]], linear_bounds=[1]),
                      dict(local_center=[0, 0], local_radius=0),
                      dict(norm_limits=[(np.eye(2), [0], 1)]),
                      dict(norm_limits=[(np.eye(2), [0, 0], np.inf)])):
            with self.assertRaises(ValueError):
                project([0, 0], [1, 0], 1, **extra)

    def test_false_solver_success_is_rejected(self):
        # A feasible point that is not the minimum must fail independent KKT.
        with patch("scipy.optimize.minimize", return_value=SimpleNamespace(
                success=True, x=np.array([.9, 0.]))):
            with self.assertRaisesRegex(ValueError, "optimum not verified"):
                project([0, 1], [1, 0], .8, local_center=[0, 0], local_radius=1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
