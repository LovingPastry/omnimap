"""Bridge fault injection without a master; uses real generated ROS messages."""
import importlib.util
import sys
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import rospy
import tf2_ros
from geometry_msgs.msg import TransformStamped, TwistStamped
from std_msgs.msg import Int8

path = Path(__file__).resolve().parents[1] / "ros_ws/src/omnimap_kinematics/scripts/camera_twist_bridge.py"
spec = importlib.util.spec_from_file_location("camera_bridge", path)
bridge = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = bridge
spec.loader.exec_module(bridge)


class BridgeTests(unittest.TestCase):
    def setUp(self):
        for target, options in [
                ("rospy.Time.now", dict(return_value=rospy.Time(100))),
                ("time.monotonic", dict(return_value=100.0)),
                ("rospy.get_param", dict(side_effect=lambda key, default: default)),
                ("rospy.Publisher", {}), ("rospy.Subscriber", {}),
                ("rospy.resolve_name", dict(side_effect=lambda name: name)),
                ("rospy.on_shutdown", {}), ("tf2_ros.Buffer", {}),
                ("tf2_ros.TransformListener", {}), ("threading.Thread.start", {}),
                ("rospy.loginfo", {})]:
            patcher = patch(target, **options)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.node = bridge.CameraTwistBridge()
        self.command = TwistStamped()
        self.command.header.stamp, self.command.header.frame_id = rospy.Time(100), "base_link"
        self.command.twist.angular.z = 2
        self.command.twist.linear.x = 0.3
        self.node.input_cb(self.command)
        self.node.status_cb(Int8(0))
        self.transforms = []
        for x in (0.2, 0.1):
            tr = TransformStamped()
            tr.header.stamp = rospy.Time(100)
            tr.transform.translation.x = x
            self.transforms.append(tr)
        self.node.buffer.lookup_transform.side_effect = lambda base, frame, stamp, timeout: (
            self.transforms[0] if frame == self.node.camera else self.transforms[1])

    def assert_zero(self, record):
        self.assertEqual(bridge.components(record.output), (0,)*6)

    def test_reference_point_and_aligned_tf_queries(self):
        record = self.node.build()
        self.assertEqual(bridge.components(record.output), (0.3, -0.2, 0, 0, 0, 2))
        self.assertTrue(record.transform_chain_valid)
        self.assertEqual(record.input, self.command)
        for call in self.node.buffer.lookup_transform.call_args_list:
            self.assertEqual(call.args[2], self.command.header.stamp)
            self.assertEqual(call.args[3], rospy.Duration(0))

    def test_status_subscription_uses_effective_servo_config_or_explicit_override(self):
        params = {"~servo_ns": "/custom/servo", "/custom/servo/status_topic": "feedback"}
        with patch("rospy.get_param", side_effect=lambda key, default: params.get(key, default)):
            node = bridge.CameraTwistBridge()
            self.assertEqual(node.status_topic, "/custom/servo/feedback")
            self.assertEqual(rospy.Subscriber.call_args.args[0], "/custom/servo/feedback")
            params["~status_topic"] = "/override/status"
            node = bridge.CameraTwistBridge()
            self.assertEqual(rospy.Subscriber.call_args.args[0], "/override/status")

    def test_ros_clock_pause_receipt_timeout_and_no_tf_block(self):
        # ROS time is unchanged, wall time has advanced; stale must still stop.
        with patch("time.monotonic", return_value=101):
            self.assert_zero(self.node.build())
        self.node.input_wall = 100
        self.node.status_wall = 99
        self.assert_zero(self.node.build())
        self.node.status_wall = 100
        self.node.buffer.lookup_transform.side_effect = tf2_ros.LookupException("missing")
        record = self.node.build()
        self.assert_zero(record)
        self.assertFalse(record.transform_chain_valid)

    def test_all_moveit_failure_codes_and_recovery(self):
        for code in (0, 1, 2, 3, 4, 5, -1, 99, 0):
            self.node.status_cb(Int8(code))
            record = self.node.build()
            if code in (0, 1):
                self.assertEqual(record.reason, "ok")
                self.assertNotEqual(bridge.components(record.output), (0,)*6)
            else:
                self.assert_zero(record)
                self.assertTrue(record.reason.startswith("moveit_"))

    def test_invalid_input_replaces_previous_motion_and_tf_age(self):
        for kind in ("frame", "nan", "old", "future", "zero_stamp"):
            cmd = TwistStamped()
            cmd.header.stamp, cmd.header.frame_id = rospy.Time(100), "base_link"
            cmd.twist.linear.x = 1
            if kind == "frame": cmd.header.frame_id = "camera"
            if kind == "nan": cmd.twist.linear.x = float("nan")
            if kind == "old": cmd.header.stamp = rospy.Time(90)
            if kind == "future": cmd.header.stamp = rospy.Time(101)
            if kind == "zero_stamp": cmd.header.stamp = rospy.Time()
            self.node.input_cb(cmd)
            self.assert_zero(self.node.build())
        self.node.input_cb(self.command)
        self.transforms[0].header.stamp = rospy.Time(99)
        record = self.node.build()
        self.assert_zero(record)
        self.assertEqual(record.reason, "tf_stale")

    def test_kalman_never_smooths_a_stop_or_reuses_failed_state(self):
        self.node.filter_mode = "kalman"
        self.assertNotEqual(bridge.components(self.node.build().output), (0,)*6)
        zero = TwistStamped()
        zero.header = self.command.header
        self.node.input_cb(zero)
        self.assert_zero(self.node.build())
        self.node.input_cb(self.command)
        self.node.build()
        self.node.status_cb(Int8(2))
        self.assert_zero(self.node.build())
        self.assertIsNone(self.node.filtered)
        self.node.status_cb(Int8(0))
        self.assertEqual(bridge.components(self.node.build().output), (0.3, -0.2, 0, 0, 0, 2))

    def test_shutdown_cannot_be_followed_by_nonzero_publication(self):
        self.node.pub = Mock()
        self.node.shutdown()
        self.node.publish()
        self.assertEqual(self.node.pub.publish.call_count, 1)
        self.assertEqual(bridge.components(self.node.pub.publish.call_args.args[0]), (0,)*6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
