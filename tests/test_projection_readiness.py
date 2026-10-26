"""Recovery readiness is independent of a previously valid local model."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"info_flow"))
from projection_readiness import ProjectionReadinessGate
from test_kinematic_feedback import feedback


class ReadinessTests(unittest.TestCase):
    def setUp(self):
        self.gate = ProjectionReadinessGate()

    def update(self, now, **fields):
        self.gate.update(feedback(stamp=now, **fields), ros_now=now, wall_now=now)

    def decision(self, now):
        return self.gate.decision(ros_now=now, wall_now=now)

    def warm(self):
        self.update(100)
        self.assertEqual(self.decision(100).state, "recovering")
        self.update(100.11)
        self.assertTrue(self.decision(100.11).ready)

    def test_missing_and_single_sample_cannot_arm(self):
        self.assertEqual(self.decision(100).state, "blocked")
        self.update(100)
        self.assertFalse(self.decision(100.15).ready)
        self.update(100.16)
        self.assertTrue(self.decision(100.16).ready)

    def test_each_stop_code_and_invalid_chain_disarms_immediately(self):
        for fields in [dict(moveit_status=code) for code in (2, 3, 4, 5, 99)]+[
                dict(transform_chain_valid=False), dict(sigma_min=float("nan"))]:
            self.gate = ProjectionReadinessGate()
            self.warm()
            self.update(100.12, **fields)
            self.assertEqual(self.decision(100.12).state, "blocked")
            self.update(100.13, moveit_status=1)
            self.assertFalse(self.decision(100.13).ready)
            self.update(100.24)
            self.assertTrue(self.decision(100.24).ready)

    def test_healthy_replay_before_newer_stop_cannot_restart_recovery(self):
        self.warm()
        self.update(100.12, moveit_status=2)
        self.gate.update(feedback(stamp=100.115), ros_now=100.13, wall_now=100.13)
        self.assertEqual(self.decision(100.13).reason, "projection_status_stamp_did_not_advance")
        self.update(100.14)
        self.assertFalse(self.decision(100.14).ready)
        self.update(100.25)
        self.assertTrue(self.decision(100.25).ready)

    def test_duplicate_stamp_resets_recovery(self):
        self.warm()
        self.update(100.11)
        self.assertFalse(self.decision(100.11).ready)
        self.update(100.12)
        self.assertFalse(self.decision(100.12).ready)
        self.update(100.23)
        self.assertTrue(self.decision(100.23).ready)

    def test_both_clocks_expire_and_stream_gap_restarts_hold(self):
        for ros_now, wall_now in ((100.4, 100.11), (100.11, 100.4), (99., 100.11), (100.11, 99.)):
            self.gate = ProjectionReadinessGate()
            self.warm()
            self.assertFalse(self.gate.decision(ros_now=ros_now, wall_now=wall_now).ready)
        self.gate = ProjectionReadinessGate()
        self.warm()
        self.update(100.4)
        self.assertFalse(self.decision(100.4).ready)
        self.update(100.51)
        self.assertTrue(self.decision(100.51).ready)

    def test_owned_feedback_and_future_stamp_not_poisoning_baseline(self):
        self.warm()
        bad = feedback(stamp=200)
        self.gate.update(bad, ros_now=100.12, wall_now=100.12)
        self.assertFalse(self.decision(100.12).ready)
        msg = feedback(stamp=100.13)
        self.gate.update(msg, ros_now=100.13, wall_now=100.13)
        msg.moveit_status = 4
        self.update(100.24)
        self.assertTrue(self.decision(100.24).ready)


if __name__ == "__main__":
    unittest.main(verbosity=2)
