"""No ROS or robot: validate proposal provenance, time and local geometry."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "info_flow"))
from kinematic_constraint import KinematicConstraintGate


def stamp(value):
    return NS(to_sec=lambda: float(value))


def command(**changes):
    fields = dict(header=NS(stamp=stamp(99.9), frame_id="base_link"), planner_tick=42,
                  model_version=7, should_stop=False, theta_rate=0.1, phi_rate=-0.2,
                  reference_radius=0.6, reference_scene_center=[0, 0, 0])
    fields.update(changes)
    return NS(**fields)


def constraint(**changes):
    fields = dict(header=NS(stamp=stamp(100), frame_id="base_link"), joint_stamp=stamp(99.99),
                  twist_stamp=stamp(99.98), command_stamp=stamp(99.9), planner_tick=42,
                  model_version=7, source_frame="base_link", task_frame="tool0",
                  valid=True, reason="ok", threshold_configured=True,
                  reference_radius=0.6, reference_scene_center=[0, 0, 0],
                  rate_scales=[0.1, 0.1], linearization_rates=[0.1, -0.2],
                  a=[1, 0], c=1.05, sigma_min=0.5, m_min=1.55, alpha=1., drift=0.,
                  inverse_residual=1e-14, derivative_error=1e-8,
                  joint_names=[f"joint_{i}" for i in range(6)],
                  joint_rate_matrix=[0.] * 12, joint_rate_bias=[0.] * 6,
                  joint_velocity_lower=[-1.] * 6, joint_velocity_upper=[1.] * 6,
                  camera_linear_matrix=[0.] * 6, camera_linear_bias=[0.] * 3,
                  previous_linear=[0.] * 3, linear_speed_limit=0.,
                  linear_step_limit=0., linear_deadband=0., servo_dt_sec=.02,
                  joint_position=[0.]*6, joint_velocity=[0.]*6,
                  joint_position_bounded=[False]*6, joint_acceleration_bounded=[False]*6,
                  joint_position_lower=[0.]*6, joint_position_upper=[0.]*6,
                  joint_acceleration_lower=[0.]*6, joint_acceleration_upper=[0.]*6)
    fields.update(changes)
    return NS(**fields)


class ConstraintTests(unittest.TestCase):
    def setUp(self):
        self.gate = KinematicConstraintGate()
        self.gate.remember_command(command())
        self.gate.update(constraint(), wall_now=100)

    def propose(self, **changes):
        args = dict(rates=[0.1, -0.2], reference_radius=.6, reference_scene_center=[0, 0, 0],
                    model_version=7, should_stop=False, state_allows_motion=True, ros_now=100, wall_now=100)
        args.update(changes)
        return self.gate.propose(**args)

    def test_projection_units_and_minimal_distance(self):
        result = self.propose()
        self.assertTrue(result.valid, result.reason)
        np.testing.assert_allclose(result.rates, [.105, -.2])
        self.assertAlmostEqual(result.correction_norm, .05)
        self.assertEqual(result.source_tick, 42)
        self.assertEqual(self.gate.command["rates"], (.1, -.2))

    def test_missing_threshold_and_upstream_failure(self):
        for fields, reason in ((dict(threshold_configured=False, c=np.nan), "threshold_unset"),
                               (dict(valid=False, reason="tf_missing"), "upstream_invalid:tf_missing")):
            self.gate.update(constraint(**fields), wall_now=100)
            self.assertIn(reason, self.propose().reason)
        self.gate.latest = None
        self.assertEqual(self.propose().reason, "constraint_missing")

    def test_source_and_receipt_age_are_independent(self):
        self.assertIn("stale", self.propose(ros_now=100.4).reason)
        self.assertIn("stale", self.propose(wall_now=100.4).reason)
        # Replayed old sample received now is still old in ROS time.
        self.gate.update(constraint(joint_stamp=stamp(99)), wall_now=100)
        self.assertIn("stale", self.propose().reason)
        for fields in (dict(header=NS(stamp=stamp(0), frame_id="base_link")),
                       dict(twist_stamp=stamp(float("nan")))):
            self.gate.update(constraint(**fields), wall_now=100)
            self.assertIn("invalid_time", self.propose().reason)
        self.gate.update(constraint(twist_stamp=stamp(100.2)), wall_now=100)
        self.assertIn("future", self.propose().reason)
        self.gate.update(constraint(joint_stamp=stamp(99.8), twist_stamp=stamp(100)), wall_now=100)
        self.assertIn("sample_skew", self.propose().reason)

    def test_provenance_and_new_stop_invalidate_old_model(self):
        for fields in (dict(planner_tick=41), dict(model_version=6), dict(command_stamp=stamp(99.8))):
            self.gate.update(constraint(**fields), wall_now=100)
            self.assertIn("unmatched", self.propose().reason)
        self.gate.update(constraint(), wall_now=100)
        self.gate.remember_command(command(should_stop=True))
        self.assertIn("stopped", self.propose().reason)
        self.gate.remember_command(command(planner_tick=43))
        self.assertIn("unmatched", self.propose().reason)

    def test_frames_map_reference_and_rates_must_match(self):
        cases = [(dict(source_frame="camera"), "frame_mismatch"),
                 (dict(task_frame="wrist"), "frame_mismatch"),
                 (dict(linearization_rates=[.11, -.2]), "rates_mismatch"),
                 (dict(reference_radius=.61), "sphere_mismatch"),
                 (dict(reference_scene_center=[.01, 0, 0]), "sphere_mismatch")]
        for fields, reason in cases:
            self.gate.update(constraint(**fields), wall_now=100)
            self.assertIn(reason, self.propose().reason)
        self.gate.update(constraint(), wall_now=100)
        self.assertIn("version_mismatch", self.propose(model_version=8).reason)
        self.assertIn("sphere_mismatch", self.propose(reference_radius=.7).reason)

    def test_bad_metrics_bound_and_unusable_direction(self):
        for fields in (dict(a=[np.nan, 0]), dict(rate_scales=[0, .1]),
                       dict(inverse_residual=.1), dict(alpha=0), dict(c=2)):
            self.gate.update(constraint(**fields), wall_now=100)
            self.assertFalse(self.propose().valid)
        self.gate.update(constraint(a=[0, 0]), wall_now=100)
        self.assertIn("projection_unavailable", self.propose().reason)

    def test_both_candidate_and_projected_point_stay_local(self):
        self.assertIn("candidate_outside", self.propose(rates=[.2, -.2]).reason)
        self.gate.update(constraint(a=[.1, 0]), wall_now=100)
        self.assertIn("projection_unavailable", self.propose().reason)

    def test_stops_cannot_be_turned_into_escape_commands(self):
        self.assertIn("stop", self.propose(should_stop=True).reason)
        self.assertIn("state_unavailable", self.propose(state_allows_motion=False).reason)

    def test_joint_local_projection_recovers_feasible_advice(self):
        # Candidate on the local ball; the plain half-space projection is
        # outside it. The circle/half-space intersection is still feasible.
        self.gate.update(constraint(c=1.2, m_min=1.7), wall_now=100)
        result = self.propose(rates=[.1, -.175])
        self.assertTrue(result.valid, result.reason)
        np.testing.assert_allclose(result.rates, [.12, -.185], atol=1e-8)

    def test_joint_velocity_intersection_and_invalid_bounds(self):
        matrix = np.zeros((6, 2))
        matrix[0] = [1, -1]
        self.gate.update(constraint(joint_rate_matrix=matrix.ravel(),
                                    joint_rate_bias=[-.05, 0, 0, 0, 0, 0],
                                    joint_velocity_upper=[2.95, 1, 1, 1, 1, 1]), wall_now=100)
        result = self.propose()
        self.assertTrue(result.valid, result.reason)
        np.testing.assert_allclose(result.rates, [.105, -.195], atol=1e-8)
        for fields in (dict(joint_names=["same"]*6), dict(joint_rate_matrix=[0]*11),
                       dict(joint_velocity_upper=[0]*6), dict(joint_velocity_lower=[np.nan]*6)):
            self.gate.update(constraint(**fields), wall_now=100)
            self.assertFalse(self.propose().valid)
        matrix[0] = [1, 0]
        self.gate.update(constraint(joint_rate_matrix=matrix.ravel(),
                                    joint_velocity_upper=[1.01, 1, 1, 1, 1, 1]), wall_now=100)
        self.assertIn("projection_unavailable", self.propose().reason)

    def test_camera_speed_and_slew_use_camera_reference_affine_model(self):
        fields = dict(camera_linear_matrix=[.1, 0, 0, .1, 0, 0],
                      camera_linear_bias=[0, .2, 0], previous_linear=[.1, 0, 0])
        for limits, y in ((dict(linear_speed_limit=.106), np.sqrt(.106**2-.105**2)),
                          (dict(linear_step_limit=.006), np.sqrt(.006**2-.005**2))):
            self.gate.update(constraint(**fields, **limits), wall_now=100)
            result = self.propose(rates=[.1, -.18])
            self.assertTrue(result.valid, result.reason)
            np.testing.assert_allclose(result.rates, [.105, -.2+y], atol=1e-8)
        self.gate.update(constraint(**fields, linear_speed_limit=.102), wall_now=100)
        self.assertIn("projection_unavailable", self.propose().reason)

    def test_deadband_supporting_halfspace_and_invalid_limits(self):
        fields = dict(camera_linear_matrix=[.1, 0, 0, 0, 0, 0], linear_deadband=.09,
                      a=[-1, 0], c=-.95, m_min=.5, drift=.95)
        self.gate.update(constraint(**fields), wall_now=100)
        result = self.propose(rates=[.08, -.2])
        self.assertTrue(result.valid, result.reason)
        np.testing.assert_allclose(result.rates, [.09, -.2], atol=1e-8)
        for changes in (dict(linear_deadband=.11), dict(linear_step_limit=-1),
                        dict(previous_linear=[np.nan]*3), dict(camera_linear_matrix=[0]*5)):
            self.gate.update(constraint(**{**fields, **changes}), wall_now=100)
            self.assertFalse(self.propose().valid)

    def test_update_owns_sample_and_explicit_recovery(self):
        msg = constraint()
        self.gate.update(msg, wall_now=100)
        msg.a[0] = 100
        self.assertAlmostEqual(self.propose().rates[0], .105)
        self.gate.update(constraint(valid=False), wall_now=100)
        self.assertFalse(self.propose().valid)
        self.gate.update(constraint(), wall_now=100)
        self.assertTrue(self.propose().valid)

    def test_joint_cycle_bounds(self):
        common = dict(a=[0., 0.], c=0., m_min=.5,
                      joint_rate_matrix=[1., 0.]+[0.]*10)
        self.gate.update(constraint(**common, joint_position=[.9802]+[0.]*5,
            joint_position_bounded=[True]+[False]*5,
            joint_position_lower=[-1.]*6, joint_position_upper=[1.]*6), wall_now=100)
        result = self.propose()
        self.assertTrue(result.valid, result.reason)
        self.assertAlmostEqual(result.rates[0], .099)
        self.gate.update(constraint(**common, joint_velocity=[.98]+[0.]*5,
            joint_acceleration_bounded=[True]*6,
            joint_acceleration_lower=[-.5]*6, joint_acceleration_upper=[.5]*6), wall_now=100)
        result = self.propose()
        self.assertTrue(result.valid, result.reason)
        self.assertAlmostEqual(result.rates[0], .099)
        self.assertTrue(result.acceleration_limits_complete)
        self.gate.update(constraint(servo_dt_sec=0), wall_now=100)
        self.assertEqual(self.propose().reason, "constraint_joint_cycle_invalid")

    def test_reject_invalid_configuration(self):
        for fields in (dict(timeout_sec=0), dict(max_sample_skew_sec=np.nan),
                       dict(max_normalized_delta=-1), dict(source_frame="/"), dict(task_frame="")):
            with self.assertRaises(ValueError):
                KinematicConstraintGate(**fields)


if __name__ == "__main__":
    unittest.main(verbosity=2)
