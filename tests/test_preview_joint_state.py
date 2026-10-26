"""Frozen-model corroboration uses names, both clocks and a healthy latest stream."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"info_flow"))
from preview_joint_state import PreviewJointState


def stamp(value):
    return NS(to_sec=lambda: value)


def sample(value=100):
    return NS(header=NS(stamp=stamp(value)), name=["j%d" % i for i in range(6)],
              position=list(range(6)), velocity=[.01*i for i in range(6)])


def response():
    msg = sample(99.99)
    return NS(joint_stamp=msg.header.stamp, joint_names=msg.name[::-1],
              joint_position=msg.position[::-1], joint_velocity=msg.velocity[::-1])


class JointTests(unittest.TestCase):
    def setUp(self):
        self.gate = PreviewJointState()
        self.gate.update(sample(), 200)

    def check(self, **kw):
        args = dict(ros_now=100, wall_now=200, timeout=.2)
        args.update(kw)
        self.gate.validate(response(), **args)

    def test_name_order_and_owned_input(self):
        msg = sample()
        gate = PreviewJointState()
        gate.update(msg, 200)
        msg.position[0] = 9
        gate.validate(response(), ros_now=100, wall_now=200, timeout=.2)
        self.check()

    def test_independent_age_and_backward_clocks(self):
        for kw in (dict(ros_now=100.3), dict(wall_now=200.3),
                   dict(ros_now=99), dict(wall_now=199)):
            with self.assertRaisesRegex(ValueError, "stale"):
                self.check(**kw)

    def test_unhealthy_latest_sample_invalidates_old_good_sample(self):
        for field, value in (("name", ["j0"]*6), ("velocity", []),
                             ("position", [np.nan]*6)):
            msg = sample(100.01)
            setattr(msg, field, value)
            self.gate.update(msg, 200)
            with self.assertRaisesRegex(ValueError, "malformed"):
                self.check()
        self.gate.update(sample(), 200)
        with self.assertRaisesRegex(ValueError, "did not advance"):
            self.check()
        self.gate.update(sample(100.02), 200)
        self.check(ros_now=100.02)

    def test_future_stamp_does_not_poison_recovery(self):
        self.gate.update(sample(101), 200, ros_now=100)
        with self.assertRaisesRegex(ValueError, "future"):
            self.check()
        self.gate.update(sample(100.01), 200, ros_now=100.01)
        self.check(ros_now=100.01)

    def test_changed_velocity_or_position_and_response_ahead(self):
        for field in ("position", "velocity"):
            msg = sample(100.01)
            getattr(msg, field)[0] += .001
            self.gate.update(msg, 200)
            with self.assertRaisesRegex(ValueError, "changed"):
                self.check(ros_now=100.01)
            self.gate = PreviewJointState()
            self.gate.update(sample(), 200)
        result = response()
        result.joint_stamp = stamp(100.01)
        with self.assertRaisesRegex(ValueError, "precedes"):
            self.gate.validate(result, ros_now=100, wall_now=200, timeout=.2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
