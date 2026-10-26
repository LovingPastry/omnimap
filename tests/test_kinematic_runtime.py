"""Exercise real Planning/Servo callbacks and generated ROS messages offline.

Only map deserialization and Fisher evaluation are replaced. No ROS master is
started, no network publisher is created, and no robot command is sent.
Source ROS + ros_ws/devel/setup.bash, then run this file in InfoFlow.
"""

import io
import sys
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace as NS, ModuleType
from unittest.mock import Mock, patch

import numpy as np
import rospy
import tf2_ros

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "info_flow")]

from omnimap_msgs.msg import KinematicStatus, SphericalCommand
from kinematic_constraint import KinematicConstraintGate
from kinematic_feedback import KinematicFeedbackGate
from test_kinematic_feedback import feedback

# Compute-side modules need torch/CUDA. These callback checks use a fixed
# snapshot/Fisher result and never call those modules. Empty torch deliberately
# has no numerical implementation: an accidental call must fail, not be mocked
# into a passing result. This also runs in the execution-side InfoFlow env.
snapshot_stub = NS(PlannerSnapshot=object, load_planner_snapshot_file=Mock(),
                   save_planner_snapshot_file=Mock())
with patch.dict(sys.modules, {
        "planner_snapshot": snapshot_stub, "torch": ModuleType("torch"),
        "matplotlib": NS(cm=None),
        "omnimap.gaussian.renderer.nbv.motion_policy": NS(FisherMotionPolicy=object),
        "omnimap.gaussian.renderer.nbv.policy_improvements": NS(nbv_policy_from_config=None)}):
    import info_flow_planning_node as planning
import info_flow_servo_runtime as servo


class Publisher:
    def __init__(self):
        self.messages = []

    def publish(self, msg):
        self.messages.append(msg)

    def get_num_connections(self):
        return 1


def logger():
    result = Mock()
    result.lines = []
    for name in ("info", "debug", "warning"):
        getattr(result, name).side_effect = lambda fmt, *args: result.lines.append(fmt % args)
    return result


def planner(mode="enforce"):
    node = planning.InfoFlowPlanningNode.__new__(planning.InfoFlowPlanningNode)
    for name in ("pose_lock", "snapshot_ref_lock", "stats_lock", "motion_policy_lock", "kinematic_status_lock", "kinematic_constraint_lock"):
        setattr(node, name, threading.Lock())
    node.stats = planning.PlanningStats()
    node.planner_tick = 41
    node.latest_pose = NS(stamp=rospy.Time(100), wall_time=100, pose_4x4=np.eye(4))
    node.latest_snapshot_ref = None
    node.snapshot_load_failed_version = -1
    node._ensure_latest_snapshot_loaded = lambda: NS(backend=NS(), model_version=7)
    node.pose_stamp_stale_timeout_sec = node.pose_receipt_stale_timeout_sec = 0.2
    node._maybe_log_status = Mock()
    node._planner_log_throttle = Mock()
    node._update_reference_sphere_online = Mock()
    node.fixed_hemisphere_center, node.fixed_hemisphere_radius_m = None, 0.6
    node.reference_center_mode = node.reference_radius_source_current = "fixed"
    node.kinematic_feedback_mode = mode
    node.kinematic_gate = KinematicFeedbackGate(deceleration_duration_sec=0)
    node.kinematic_projection_mode = "observe"
    node.kinematic_constraint_gate = KinematicConstraintGate()
    node.world_frame = "base_link"
    node.profile_logger = logger()
    node.SphericalCommandMsg = SphericalCommand
    node.spherical_cmd_pub, node.raw_spherical_cmd_pub = Publisher(), Publisher()
    node.calib, node.image_size_hw = [500, 500, 320, 240], (480, 640)
    node.motion_policy = Mock(dt=0.02, last_timing={})
    node.motion_policy.next_pose_from_c2w.return_value = NS(
        reference_radius=0.6, reference_scene_center=[0, 0, 0], dt=0.02,
        theta_rate_applied=0.1, phi_rate_applied=-0.2, fisher_score=12,
        should_stop=False, stop_reason="fisher_motion",
    )
    return node


class PlanningTests(unittest.TestCase):
    def setUp(self):
        self.time_patch = patch.object(rospy.Time, "now", return_value=rospy.Time(100))
        self.wall_patch = patch("time.monotonic", return_value=100)
        self.time_patch.start()
        self.wall_patch.start()
        self.addCleanup(self.time_patch.stop)
        self.addCleanup(self.wall_patch.stop)

    def test_observe_preserves_fisher_and_default_mode(self):
        self.assertEqual(planning.build_argparser().parse_args([]).kinematic_feedback_mode, "observe")
        for msg in (None, feedback(moveit_status=2), feedback(transform_chain_valid=False)):
            node = planner("observe")
            if msg is not None:
                node.kinematic_status_callback(msg)
            node._planning_timer_callback(None)
            raw, output = node.raw_spherical_cmd_pub.messages[0], node.spherical_cmd_pub.messages[0]
            self.assertEqual(raw, output)
            self.assertFalse(output.should_stop)
            self.assertTrue(any("mode=observe" in line for line in node.profile_logger.lines))

    def test_projection_advice_is_logged_without_changing_control(self):
        from omnimap_msgs.msg import KinematicConstraint
        node = planner("observe")
        node.kinematic_status_callback(feedback())
        node._planning_timer_callback(None)  # establishes the actual source command
        msg = KinematicConstraint()
        msg.header.stamp = msg.joint_stamp = msg.twist_stamp = msg.command_stamp = rospy.Time(100)
        msg.header.frame_id = msg.source_frame = "base_link"
        msg.task_frame = "tool0"
        msg.planner_tick, msg.model_version = 42, 7
        msg.valid = msg.threshold_configured = True
        msg.reference_radius = .6
        msg.rate_scales, msg.linearization_rates = [.1, .1], [.1, -.2]
        msg.servo_dt_sec = .02
        msg.joint_names = [f"joint_{i}" for i in range(6)]
        msg.joint_velocity_lower, msg.joint_velocity_upper = [-1.]*6, [1.]*6
        msg.a, msg.c, msg.sigma_min, msg.m_min, msg.alpha = [1, 0], 1.05, .5, 1.55, 1.
        node.kinematic_constraint_callback(msg)
        node._planning_timer_callback(None)
        raw, output = node.raw_spherical_cmd_pub.messages[-1], node.spherical_cmd_pub.messages[-1]
        self.assertEqual(raw, output)
        self.assertEqual((output.theta_rate, output.phi_rate), (.1, -.2))
        self.assertTrue(any("reason=advisory_projection" in line and "proposed=[0.105 -0.2]" in line
                            for line in node.profile_logger.lines))
        # The same old constraint no longer refers to the latest published command.
        node._planning_timer_callback(None)
        self.assertTrue(any("constraint_execution_command_unmatched" in line for line in node.profile_logger.lines))
        self.assertEqual(planning.build_argparser().parse_args([]).kinematic_projection_mode, "observe")

    def test_enforce_stops_and_raw_remains_replayable(self):
        for msg in (None, feedback(stamp=90), feedback(task_frame="camera"),
                    feedback(transform_chain_valid=False),
                    *(feedback(moveit_status=code) for code in (2, 3, 4, 5, 99))):
            with self.subTest(msg=msg):
                node = planner()
                if msg is not None:
                    node.kinematic_status_callback(msg)
                node._planning_timer_callback(None)
                raw, output = node.raw_spherical_cmd_pub.messages[0], node.spherical_cmd_pub.messages[0]
                self.assertEqual((raw.theta_rate, raw.phi_rate), (0.1, -0.2))
                self.assertFalse(raw.should_stop)
                self.assertEqual((output.theta_rate, output.phi_rate), (0, 0))
                self.assertTrue(output.should_stop)
                self.assertEqual((raw.planner_tick, raw.header.stamp), (output.planner_tick, output.header.stamp))
                self.assertEqual(node.stats.planner_stop_kinematic, 1)

    def test_deceleration_preserves_policy_stop_and_recovers(self):
        node = planner()
        node.kinematic_status_callback(feedback(moveit_status=1))
        node._planning_timer_callback(None)
        output = node.spherical_cmd_pub.messages[-1]
        self.assertEqual((output.theta_rate, output.phi_rate), (0.05, -0.1))
        result = node.motion_policy.next_pose_from_c2w.return_value
        result.should_stop, result.stop_reason = True, "fisher_converged"
        node._planning_timer_callback(None)
        self.assertEqual(node.spherical_cmd_pub.messages[-1].stop_reason, "fisher_converged")
        result.should_stop = False
        node.kinematic_status_callback(feedback(moveit_status=0))
        node._planning_timer_callback(None)
        self.assertEqual(node.spherical_cmd_pub.messages[-1].theta_rate, 0.1)

    def test_feedback_arriving_during_fisher_is_used(self):
        node = planner()
        node.kinematic_status_callback(feedback())
        result = node.motion_policy.next_pose_from_c2w.return_value
        def evaluate(**kwargs):
            node.kinematic_status_callback(feedback(moveit_status=2))
            return result
        node.motion_policy.next_pose_from_c2w.side_effect = evaluate
        node._planning_timer_callback(None)
        self.assertTrue(node.spherical_cmd_pub.messages[-1].should_stop)

    def test_generated_status_message_roundtrip(self):
        fields = feedback().__dict__.copy()
        fields.pop("header")
        status = KinematicStatus(**fields)
        status.header.stamp = rospy.Time(100)
        buffer = io.BytesIO()
        status.serialize(buffer)
        decoded = KinematicStatus().deserialize(buffer.getvalue())
        self.assertEqual(status, decoded)
        node = planner()
        node.kinematic_status_callback(decoded)
        node._planning_timer_callback(None)
        self.assertFalse(node.spherical_cmd_pub.messages[-1].should_stop)


class ServoTests(unittest.TestCase):
    def setUp(self):
        self.time_patch = patch.object(rospy.Time, "now", return_value=rospy.Time(100))
        self.wall_patch = patch("time.monotonic", return_value=100)
        self.time_patch.start()
        self.wall_patch.start()
        self.addCleanup(self.time_patch.stop)
        self.addCleanup(self.wall_patch.stop)
        args = servo.build_argparser().parse_args([])
        for key, value in dict(servo_hz=50, pose_stale_timeout_sec=0.2,
                spherical_cmd_timeout_sec=0.25, linear_vel_max=0.1, linear_deadband=0,
                angular_speed_max=1, enable_angular=False, radial_gain=0, radial_i_gain=0,
                radial_deadband=0.001, radial_integral_limit=0.05, radial_vel_max=0.015,
                angular_gain=1, angular_deadband=0.001, linear_accel_max=0,
                angular_accel_max=0).items():
            setattr(args, key, value)
        self.pub = Publisher()
        self.trace_pub = Publisher()
        self.log = logger()
        with patch.object(rospy, "init_node"), patch.object(rospy, "Publisher", side_effect=[self.pub, self.trace_pub]), \
                patch.object(rospy, "Subscriber"), patch.object(rospy, "Timer"), \
                patch.object(tf2_ros, "Buffer"), patch.object(tf2_ros, "TransformListener"), \
                patch.object(servo, "get_section_logger", return_value=self.log):
            self.node = servo.InfoFlowServoRuntime(args)
        self.node._maybe_log_status = Mock()
        self.cmd = SphericalCommand(planner_tick=42, model_version=7, theta_rate=0.1,
                                    reference_radius=0.6, reference_scene_center=[0, 0, 0])
        self.cmd.header.stamp, self.cmd.header.frame_id = rospy.Time(100), "base_link"
        self.node.spherical_cmd_callback(self.cmd)

    def assert_zero(self):
        msg = self.pub.messages[-1]
        np.testing.assert_array_equal([msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z,
                                      msg.twist.angular.x, msg.twist.angular.y, msg.twist.angular.z], np.zeros(6))

    def test_stop_needs_no_tf_and_is_traced(self):
        self.cmd.should_stop, self.cmd.stop_reason = True, "moveit_halt_for_singularity:2"
        with patch.object(servo, "lookup_latest_pose_from_tf") as lookup:
            self.node._servo_timer_callback(None)
            lookup.assert_not_called()
        self.assert_zero()
        self.assertIn("tick=42", self.log.lines[-1])
        self.assertIn("reason=moveit_halt_for_singularity:2", self.log.lines[-1])

    def test_frame_tf_failure_and_age_gates(self):
        self.cmd.header.frame_id = "camera"
        self.node._servo_timer_callback(None)
        self.assert_zero()
        self.cmd.header.frame_id = "base_link"
        with patch.object(servo, "lookup_latest_pose_from_tf", side_effect=tf2_ros.LookupException("missing")):
            self.node._servo_timer_callback(None)
        self.assert_zero()
        pose = np.eye(4)
        pose[0, 3] = 0.6
        for stamp in (rospy.Time(99), rospy.Time(101)):
            with patch.object(servo, "lookup_latest_pose_from_tf", return_value=(pose, stamp)):
                self.node._servo_timer_callback(None)
            self.assert_zero()

    def test_twist_values_match_trace_without_reference_point_transform(self):
        pose = np.eye(4)
        pose[0, 3] = 0.6
        with patch.object(servo, "lookup_latest_pose_from_tf", return_value=(pose, rospy.Time(100))):
            self.node._servo_timer_callback(None)
        msg = self.pub.messages[-1]
        np.testing.assert_allclose([msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z], [0, 0.06, 0])
        self.assertEqual(msg.header.frame_id, "base_link")
        trace = next(line for line in self.log.lines if line.startswith("servo_twist:"))
        self.assertIn("linear=[0 0.06 0]", trace)
        self.assertIn("tick=42 model_v=7", trace)
        self.assertIn("twist_stamp=100.000000000", trace)
        companion = self.trace_pub.messages[-1]
        self.assertEqual(companion.planner_tick, 42)
        self.assertEqual(companion.model_version, 7)
        self.assertEqual(companion.command, msg)
        self.assertTrue(companion.rate_model_valid, companion.rate_model_reason)
        self.assertTrue(companion.linear_limits_valid)
        np.testing.assert_array_equal(companion.previous_linear, [0, 0, 0])
        self.assertEqual(companion.linear_speed_limit, .1)
        self.assertEqual(companion.linear_step_limit, 0.)
        matrix = np.array(companion.camera_rate_matrix).reshape(6, 2)
        np.testing.assert_allclose(matrix @ companion.spherical_rates + companion.camera_bias,
                                   [msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z,
                                    msg.twist.angular.x, msg.twist.angular.y, msg.twist.angular.z])
        self.assertEqual(companion.reference_radius, self.cmd.reference_radius)
        np.testing.assert_array_equal(companion.reference_scene_center, self.cmd.reference_scene_center)
        buffer = io.BytesIO()
        companion.serialize(buffer)
        self.assertEqual(type(companion)().deserialize(buffer.getvalue()), companion)

    def test_limit_trace_uses_previous_command_and_actual_cycle_step(self):
        self.node._last_linear_cmd = np.array([0., .05, 0.])
        self.node.linear_accel_max = 2.
        pose = np.eye(4)
        pose[0, 3] = .6
        with patch.object(servo, "lookup_latest_pose_from_tf", return_value=(pose, rospy.Time(100))), \
                patch.object(self.node, "_servo_dt_sec", return_value=.03):
            self.node._servo_timer_callback(None)
        trace = self.trace_pub.messages[-1]
        self.assertTrue(trace.rate_model_valid, trace.rate_model_reason)
        np.testing.assert_array_equal(trace.previous_linear, [0., .05, 0.])
        np.testing.assert_allclose(self.node._last_linear_cmd, [0., .06, 0.])
        self.assertEqual(trace.linear_step_limit, .06)
        self.assertEqual(trace.servo_dt_sec, .03)

    def test_preview_query_is_repeatable_and_does_not_commit_or_publish(self):
        pose = np.eye(4)
        pose[0, 3] = .5
        self.node.enable_angular, self.node.angular_accel_max = True, .5
        self.node._last_angular_cmd = np.array([.01, 0., 0.])
        self.node.radial_gain, self.node.radial_i_gain = .01, .02
        self.node._radial_integral_error = .01
        self.node._last_linear_cmd = np.array([.001, .04, 0.])
        self.cmd.should_stop = True
        old_linear = self.node._last_linear_cmd.copy()
        old_angular = self.node._last_angular_cmd.copy()
        response = object()
        query = Mock(return_value=NS(constraint=response))
        for _ in range(2):
            self.assertIs(self.node.evaluate_preview(
                self.cmd, pose, rospy.Time(100), .02, query), response)
        first, second = [call.args[0] for call in query.call_args_list]
        self.assertEqual(first, second)
        self.assertEqual(first.reason, "preview")
        self.assertEqual(first.servo_dt_sec, .02)
        self.assertEqual(self.node._radial_integral_error, .01)
        np.testing.assert_array_equal(self.node._last_linear_cmd, old_linear)
        np.testing.assert_array_equal(self.node._last_angular_cmd, old_angular)
        self.assertEqual(self.pub.messages, [])
        self.assertEqual(self.trace_pub.messages, [])
        # An actual fresh cycle from the same controller state yields the same
        # Twist, and only this cycle commits the radial integral/slew state.
        self.cmd.should_stop = False
        with patch.object(servo, "lookup_latest_pose_from_tf", return_value=(pose, rospy.Time(100))), \
                patch.object(self.node, "_servo_dt_sec", return_value=.02):
            self.node._servo_timer_callback(None)
        self.assertEqual(first.command.twist, self.pub.messages[-1].twist)
        self.assertGreater(self.node._radial_integral_error, .01)

    def test_preview_rejects_stale_frame_and_nonaffine_output_before_query(self):
        pose = np.eye(4)
        pose[0, 3] = .6
        query = Mock()
        for stamp in (rospy.Time(99), rospy.Time(101)):
            with self.assertRaisesRegex(ValueError, "stale"):
                self.node.evaluate_preview(self.cmd, pose, stamp, .02, query)
        self.cmd.header.frame_id = "tool0"
        with self.assertRaisesRegex(ValueError, "frame"):
            self.node.evaluate_preview(self.cmd, pose, rospy.Time(100), .02, query)
        self.cmd.header.frame_id = "base_link"
        self.node.linear_accel_max = .001
        with self.assertRaisesRegex(ValueError, "linear_slew_limit_active"):
            self.node.evaluate_preview(self.cmd, pose, rospy.Time(100), .02, query)
        query.assert_not_called()
        self.assertEqual(self.pub.messages, [])

    def test_missing_stale_and_nonfinite_commands_stop(self):
        cached = self.node.latest_cmd
        self.node.latest_cmd = None
        self.node._servo_timer_callback(None)
        self.assert_zero()
        self.node.latest_cmd = cached
        cached.receipt_wall_time = 99
        self.node._servo_timer_callback(None)
        self.assert_zero()
        cached.receipt_wall_time = 100
        self.cmd.theta_rate = float("nan")
        pose = np.eye(4)
        pose[0, 3] = 0.6
        with patch.object(servo, "lookup_latest_pose_from_tf", return_value=(pose, rospy.Time(100))):
            self.node._servo_timer_callback(None)
        self.assert_zero()
        self.assertIn("servo_exception:nonfinite spherical command", self.log.lines[-1])


if __name__ == "__main__":
    unittest.main(verbosity=2)
