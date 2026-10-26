"""Stream replay/restart, ownership and fault epoch behavior."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"info_flow"))
from kinematic_model import KinematicModelCache


def model(seq=1, stamp=100., valid=True):
    return NS(header=NS(stamp=NS(to_sec=lambda: stamp)), model_sequence=seq,
              valid=valid, reason="synthetic", data=[1])


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.cache = KinematicModelCache()

    def update(self, msg):
        return self.cache.update(msg, ros_now=100, wall_now=200, timeout_sec=.2)

    def get(self, **kw):
        args = dict(ros_now=100, wall_now=200, timeout_sec=.2)
        args.update(kw)
        return self.cache.get(**args)

    def test_owned_sample_and_dual_clock_age(self):
        source = model()
        self.assertTrue(self.update(source))
        source.data[0] = 2
        owned, _, epoch = self.get()
        self.assertEqual(owned.data, [1])
        owned.data[0] = 3
        self.assertEqual(self.get()[0].data, [1])
        self.assertTrue(self.cache.epoch_matches(epoch))
        for kw in (dict(ros_now=100.3), dict(wall_now=200.3), dict(ros_now=99), dict(wall_now=199)):
            with self.assertRaisesRegex(ValueError, "stale"):
                self.get(**kw)

    def test_replay_and_invalid_sample_clear_old_result(self):
        self.update(model())
        epoch = self.get()[2]
        self.assertFalse(self.update(model()))
        self.assertFalse(self.cache.epoch_matches(epoch))
        with self.assertRaisesRegex(ValueError, "replay"):
            self.get()
        self.assertTrue(self.update(model(2, 100.01)))
        epoch = self.get()[2]
        self.assertFalse(self.update(model(3, 100.02, False)))
        self.assertTrue(self.update(model(4, 100.03)))
        self.assertFalse(self.cache.epoch_matches(epoch))

    def test_restart_requires_next_increasing_sample(self):
        self.update(model(10))
        self.assertFalse(self.update(model(1, 100.01)))
        with self.assertRaises(ValueError):
            self.get()
        self.assertTrue(self.update(model(2, 100.02)))
        self.assertEqual(self.get()[0].model_sequence, 2)

    def test_future_or_old_stamp_cannot_poison_baseline(self):
        self.update(model())
        self.assertFalse(self.update(model(100, 200)))
        self.assertTrue(self.update(model(2, 100.01)))
        self.assertFalse(self.update(model(100, 99.99)))
        self.assertTrue(self.update(model(3, 100.02)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
