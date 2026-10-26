#!/usr/bin/env python3
"""Real ROS nodes + MoveIt RobotState, synthetic chain, no hardware topics.

Independent NumPy forward kinematics and finite differences check the monitor's
frame, reference point, variable order, normalization and measured speed.
"""
import copy
import threading
import time
import unittest
import xml.etree.ElementTree as ET

import numpy as np
import rospy
import rostest
import tf2_ros
from geometry_msgs.msg import TransformStamped, TwistStamped
from omnimap_msgs.msg import BridgeStatus, KinematicStatus, ServoTwistTrace, KinematicConstraint
from omnimap_msgs.srv import EvaluateKinematicModel
from sensor_msgs.msg import JointState
from std_msgs.msg import Int8
from rosgraph_msgs.msg import Clock
from roscpp.srv import GetLoggers
from tf.transformations import euler_matrix, rotation_matrix, quaternion_from_matrix


def values(msg):
    v, w = msg.twist.linear, msg.twist.angular
    return np.array([v.x, v.y, v.z, w.x, w.y, w.z])


class ChainTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rospy.init_node("test_kinematic_chain")
        if rospy.get_param("/test/configure_servo_late", False):
            service = "/omnimap_kinematic_monitor/get_loggers"
            rospy.wait_for_service(service, timeout=5)
            probe = rospy.ServiceProxy(service, GetLoggers)
            probe()
            time.sleep(0.3)  # wall time: simulated /clock is not started yet
            probe()  # monitor must survive missing Servo config and paused time
            rospy.set_param("/test/servo", {
                "move_group_name": "manipulator", "planning_frame": "base_link",
                "robot_link_command_frame": "base_link", "ee_frame_name": "tool0",
                "joint_topic": "/test/joints", "status_topic": "feedback",
                "cartesian_command_in_topic": "/test/tool_twist"})
        cls.lock = threading.RLock()
        cls.kinematic, cls.bridge, cls.output, cls.constraint = None, None, None, None
        cls.subs = [rospy.Subscriber(topic, kind, lambda msg, key=key: setattr(cls, key, msg))
                    for topic, kind, key in [("/test/kinematic", KinematicStatus, "kinematic"),
                                      ("/test/constraint", KinematicConstraint, "constraint"),
                        ("/test/bridge", BridgeStatus, "bridge"), ("/test/tool_twist", TwistStamped, "output")]]
        cls.joint_pub = rospy.Publisher("/test/joints", JointState, queue_size=1)
        cls.status_pub = rospy.Publisher("/test/servo/feedback", Int8, queue_size=1)
        cls.input_pub = rospy.Publisher("/test/camera_twist", TwistStamped, queue_size=1)
        cls.trace_pub = rospy.Publisher("/test/trace", ServoTwistTrace, queue_size=1)
        cls.clock_pub = rospy.Publisher("/clock", Clock, queue_size=1)
        cls.tf = tf2_ros.TransformBroadcaster()
        cls.joints = ET.fromstring(rospy.get_param("/robot_description")).findall("joint")
        cls.closed = threading.Event()
        cls.worker = threading.Thread(target=cls.feed, daemon=True)
        cls.worker.start()

    @classmethod
    def tearDownClass(cls):
        cls.closed.set()
        cls.worker.join(timeout=2)

    def setUp(self):
        with self.lock:
            self.__class__.pause_clock = False
            self.__class__.q = np.array([0.2, -0.8, 1.3, 0.4, -0.7, 0.5])
            self.__class__.dq = np.array([0.01, -0.02, 0.03, 0.04, -0.05, 0.06])
            self.__class__.code = 0
            self.__class__.send_input = self.__class__.send_status = self.__class__.send_joints = True
            self.__class__.send_tf = True
            self.__class__.input_age = 0.06
            self.__class__.joint_age = None
            self.__class__.tf_age = 0.0
            self.__class__.input_frame = "base_link"
            self.__class__.malformed = False
            self.__class__.tf_error = 0.0
            self.__class__.rate_model_valid = True
            self.__class__.rate_model_error = 0.0
            self.__class__.linear_limits_valid = True
        self.wait(lambda: self.kinematic is not None and self.kinematic.transform_chain_valid,
                  "healthy chain", timeout=6)

    @classmethod
    def fk(cls, q):
        poses = {"world": np.eye(4)}
        for joint in cls.joints:
            parent, child = joint.find("parent").get("link"), joint.find("child").get("link")
            origin = joint.find("origin")
            rpy = np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")
            xyz = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
            matrix = euler_matrix(*rpy)
            matrix[:3, 3] = xyz
            if joint.get("type") == "revolute":
                index = int(joint.get("name")[1:])-1
                matrix = matrix @ rotation_matrix(q[index], np.fromstring(joint.find("axis").get("xyz"), sep=" "))
            poses[child] = poses[parent] @ matrix
        inv = np.linalg.inv(poses["base_link"])
        mount = euler_matrix(0.2, -0.1, 0.3)
        mount[:3, 3] = [0.12, 0.03, 0.08]
        poses["cam_1_color_optical_frame"] = poses["tool0"] @ mount
        return {key: inv @ value for key, value in poses.items()}

    @classmethod
    def feed(cls):
        while not cls.closed.wait(0.02) and not rospy.is_shutdown():
            with cls.lock:
                if getattr(cls, "pause_clock", False):
                    continue
                # Stamp this frame from the clock value just published, not
                # rospy.Time.now(): its asynchronous /clock callback may still
                # hold the previous tick and create duplicate joint stamps.
                now = rospy.Time.from_sec(time.time())
                cls.clock_pub.publish(Clock(now))
                if not hasattr(cls, "q"):
                    continue
                # Publish TF ahead of the input, simulating sensor/compute delay.
                poses = cls.fk(cls.q)
                if cls.send_tf:
                    transforms = []
                    for frame in ("tool0", "cam_1_color_optical_frame"):
                        tr = TransformStamped()
                        tr.header.stamp, tr.header.frame_id, tr.child_frame_id = now-rospy.Duration(cls.tf_age), "base_link", frame
                        xyz = poses[frame][:3, 3].copy()
                        xyz[0] += cls.tf_error
                        tr.transform.translation.x, tr.transform.translation.y, tr.transform.translation.z = xyz
                        quat = quaternion_from_matrix(poses[frame])
                        tr.transform.rotation.x, tr.transform.rotation.y, tr.transform.rotation.z, tr.transform.rotation.w = quat
                        transforms.append(tr)
                    cls.tf.sendTransform(transforms)
                stamp = now-rospy.Duration(cls.input_age)
                if cls.send_joints:
                    js = JointState()
                    js.header.stamp = stamp if cls.joint_age is None else now-rospy.Duration(cls.joint_age)
                    # Deliberately reverse input order relative to RobotModel.
                    js.name = ["j%d" % i for i in range(6, 0, -1)]
                    js.position, js.velocity = cls.q[::-1].tolist(), cls.dq[::-1].tolist()
                    if cls.malformed:
                        js.name[0] = "missing_joint"
                    cls.joint_pub.publish(js)
                if cls.send_status:
                    cls.status_pub.publish(Int8(cls.code))
                if cls.send_input:
                    msg = TwistStamped()
                    msg.header.stamp, msg.header.frame_id = stamp, cls.input_frame
                    msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z = 0.02, -0.01, 0.03
                    msg.twist.angular.x, msg.twist.angular.y, msg.twist.angular.z = 0.1, -0.2, 0.3
                    trace = ServoTwistTrace(planner_tick=42, model_version=7, command_stamp=stamp, command=msg)
                    trace.header.stamp = stamp
                    trace.reason = "motion"
                    trace.rate_model_valid = cls.rate_model_valid
                    trace.rate_model_reason = "test_affine" if cls.rate_model_valid else "linear_slew_limit_active"
                    trace.reference_radius = 0.5
                    trace.spherical_rates = [0.1, -0.05]
                    matrix = np.zeros((6, 2))
                    matrix[0, 0] = matrix[1, 1] = 0.2
                    trace.camera_rate_matrix = matrix.reshape(-1).tolist()
                    trace.camera_bias = [cls.rate_model_error, 0, 0.03, 0.1, -0.2, 0.3]
                    trace.servo_dt_sec = .02
                    trace.linear_limits_valid = cls.linear_limits_valid
                    trace.previous_linear = [0.01, 0, 0.03]
                    trace.linear_speed_limit, trace.linear_step_limit, trace.linear_deadband = .1, .05, .001
                    cls.input_pub.publish(msg)
                    # rospy has now assigned msg.header.seq. Nested headers in
                    # the trace preserve that actual wire sequence.
                    cls.latest_trace = copy.deepcopy(trace)
                    cls.trace_pub.publish(trace)

    def wait(self, predicate, description, timeout=3):
        deadline = time.monotonic()+timeout
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(0.02)
        self.fail("timed out: %s; monitor=%s bridge=%s" %
                  (description, self.kinematic.reason if self.kinematic else None,
                   self.bridge.reason if self.bridge else None))

    def test_metrics_frame_order_and_reference_point(self):
        q = self.q.copy()
        poses = self.fk(q)
        jac = np.empty((6, 6))
        for i in range(6):
            delta = np.zeros(6)
            delta[i] = 1e-6
            plus, minus = self.fk(q+delta)["tool0"], self.fk(q-delta)["tool0"]
            jac[:3, i] = (plus[:3, 3]-minus[:3, 3])/2e-6
            rdot = ((plus[:3, :3]-minus[:3, :3])/2e-6) @ poses["tool0"][:3, :3].T
            jac[3:, i] = [rdot[2, 1], rdot[0, 2], rdot[1, 0]]
        dx = np.arange(1, 7)*0.1
        normalized = jac/dx[:, None]*np.arange(1, 7)[None, :]
        sv = np.linalg.svd(normalized, compute_uv=False)
        self.wait(lambda: self.kinematic.transform_chain_valid and self.kinematic.planner_tick == 42, "tick correlation")
        msg = self.kinematic
        np.testing.assert_allclose([msg.sigma_min, msg.sigma_max, msg.condition_number, msg.manipulability],
                                   [sv[-1], sv[0], sv[0]/sv[-1], np.prod(sv)], rtol=1e-6)
        self.assertAlmostEqual(msg.actual_speed_norm, np.linalg.norm(jac @ self.dq/dx), places=7)
        self.assertEqual((msg.source_frame, msg.task_frame), ("base_link", "tool0"))
        offset = poses["cam_1_color_optical_frame"][:3, 3]-poses["tool0"][:3, 3]
        incoming = values(self.bridge.input)
        expected = np.r_[incoming[:3]-np.cross(incoming[3:], offset), incoming[3:]]
        np.testing.assert_allclose(values(self.bridge.output), expected, atol=1e-9)
        self.assertAlmostEqual(msg.command_speed_norm, np.linalg.norm(expected/dx), places=7)

    def test_delayed_tf_uses_bounded_joint_history(self):
        # TF consistently trails joints by 80 ms. Latest-only lookup cannot
        # succeed; a fresh historical joint sample must drive RobotState and TF.
        with self.lock:
            self.__class__.joint_age = 0.0
            self.__class__.tf_age = 0.08
            self.__class__.input_age = 0.12
        time.sleep(0.5)
        for _ in range(15):
            self.assertTrue(self.kinematic.transform_chain_valid, self.kinematic.reason)
            time.sleep(0.05)
        with self.lock:
            self.__class__.send_joints = False
        time.sleep(0.5)
        self.assertFalse(self.kinematic.transform_chain_valid)
        with self.lock:
            self.__class__.send_joints = True
        self.wait(lambda: self.kinematic.transform_chain_valid, "delayed TF recovery")

    def independent_jacobian(self, q):
        jac = np.empty((6, 6))
        rotation = self.fk(q)["tool0"][:3, :3]
        for i in range(6):
            delta = np.zeros(6)
            delta[i] = 1e-6
            plus, minus = self.fk(q+delta)["tool0"], self.fk(q-delta)["tool0"]
            jac[:3, i] = (plus[:3, 3]-minus[:3, 3])/2e-6
            rdot = ((plus[:3, :3]-minus[:3, :3])/2e-6) @ rotation.T
            jac[3:, i] = [rdot[2, 1], rdot[0, 2], rdot[1, 0]]
        return jac

    def test_preview_reacquires_model_without_fresh_bridge(self):
        service = ("/omnimap_kinematic_monitor/evaluate_model" if
                   rospy.get_param("/test/configure_servo_late", False) else
                   "/test_monitor/evaluate_model")
        rospy.wait_for_service(service, timeout=3)
        query = rospy.ServiceProxy(service, EvaluateKinematicModel)
        self.wait(lambda: hasattr(self, "latest_trace") and self.constraint is not None
                  and self.constraint.valid, "source model")
        baseline = self.constraint
        with self.lock:
            self.__class__.send_input = False
            candidate = copy.deepcopy(self.latest_trace)
        # The ordinary monitor loses its bridge source. Queries use only the
        # candidate timestamp plus fresh q/dq, TF and actual MoveIt status.
        time.sleep(.5)
        self.wait(lambda: not self.kinematic.transform_chain_valid, "bridge expires")
        def refresh():
            candidate.reason = "preview"
            candidate.command.header.stamp = rospy.Time.now()-rospy.Duration(.06)
            candidate.command_stamp = candidate.command.header.stamp
        refresh()
        result = query(candidate).constraint
        self.assertTrue(result.valid, result.reason)
        self.assertTrue(result.reason.startswith("advisory_preview:"))
        np.testing.assert_allclose(result.joint_rate_matrix, baseline.joint_rate_matrix)
        np.testing.assert_allclose(result.joint_rate_bias, baseline.joint_rate_bias)
        self.assertGreater(result.joint_stamp.to_sec(), baseline.joint_stamp.to_sec())
        # A mathematical preview may violate a limit; the service returns its
        # map and limits, while the consumer must solve feasibility. Never send
        # this candidate to a control publisher.
        candidate.linear_speed_limit = .001
        candidate.rate_model_reason = "unclipped_preview:linear_speed_limit_active"
        refresh()
        limited = query(candidate).constraint
        self.assertTrue(limited.valid, limited.reason)
        self.assertEqual(limited.linear_speed_limit, .001)
        candidate.command.header.stamp = rospy.Time(1)
        self.assertIn("preview_stale", query(candidate).constraint.reason)
        refresh()
        candidate.command.header.frame_id = "tool0"
        self.assertEqual(query(candidate).constraint.reason, "preview_reason_or_frame_invalid")
        candidate.command.header.frame_id = "base_link"
        with self.lock:
            self.__class__.code = 2
        time.sleep(.1)
        refresh()
        rejected = query(candidate).constraint
        self.assertFalse(rejected.valid)
        self.assertEqual(rejected.reason, "moveit_not_executable")

    def test_directional_model_matches_independent_fk_and_bridge(self):
        self.wait(lambda: self.constraint is not None and self.constraint.valid, "constraint derivatives")
        result = self.constraint
        q = self.q.copy()
        jac = self.independent_jacobian(q)
        poses = self.fk(q)
        offset = poses["cam_1_color_optical_frame"][:3, 3]-poses["tool0"][:3, 3]
        task = np.zeros((6, 3))
        task[0, 0], task[1, 1] = 0.2 * result.rate_scales[0], 0.2 * result.rate_scales[1]
        task[:, 2] = np.r_[np.array([0, 0, 0.03])-np.cross([0.1, -0.2, 0.3], offset), [0.1, -0.2, 0.3]]
        directions = np.linalg.solve(jac, task)
        np.testing.assert_allclose(np.asarray(result.joint_rate_matrix).reshape(6, 2),
                                   directions[:, :2], atol=1e-7)
        np.testing.assert_allclose(result.joint_rate_bias, directions[:, 2], atol=1e-7)
        self.assertEqual(result.servo_dt_sec, .02)
        self.assertTrue(all(result.joint_position_bounded))
        expected_acceleration = [rospy.get_param(
            "/robot_description_planning/joint_limits/j%d/max_acceleration" % i, 0.)
            for i in range(1, 7)]
        self.assertEqual(list(result.joint_acceleration_bounded),
                         [value > 0 for value in expected_acceleration])
        np.testing.assert_allclose(result.joint_acceleration_upper, expected_acceleration)
        np.testing.assert_allclose(result.joint_acceleration_lower, -np.array(expected_acceleration))
        np.testing.assert_allclose(result.joint_position, q)
        np.testing.assert_allclose(result.joint_velocity, self.dq)
        np.testing.assert_allclose(result.joint_position_lower, [-6.28]*6)
        np.testing.assert_allclose(result.joint_position_upper, [6.28]*6)
        np.testing.assert_allclose(result.joint_velocity_upper, np.arange(1, 7))
        np.testing.assert_allclose(result.joint_velocity_lower, -np.arange(1, 7))
        np.testing.assert_allclose(np.asarray(result.camera_linear_matrix).reshape(3, 2), task[:3, :2])
        np.testing.assert_allclose(result.camera_linear_bias, [0, 0, .03])
        np.testing.assert_allclose(result.previous_linear, [.01, 0, .03])
        self.assertEqual((result.linear_speed_limit, result.linear_step_limit, result.linear_deadband),
                         (.1, .05, .001))
        self.assertEqual(list(result.joint_names), ["j%d" % i for i in range(1, 7)])
        def metric(q):
            scaled = self.independent_jacobian(q)/((np.arange(1, 7)*0.1)[:, None]) * np.arange(1, 7)[None, :]
            return np.linalg.svd(scaled, compute_uv=False)[-1]
        expected = []
        for direction in directions.T:
            step = 2e-4/np.linalg.norm(direction)
            expected.append((metric(q+step*direction)-metric(q-step*direction))/(2*step))
        np.testing.assert_allclose([*result.a, result.drift], expected, atol=2e-5, rtol=2e-3)
        self.assertLess(result.inverse_residual, 1e-6)
        self.assertEqual(result.planner_tick, 42)
        self.assertEqual(result.model_version, 7)
        self.assertFalse(result.joint_stamp.is_zero())
        if rospy.get_param("/test/configure_servo_late", False):
            self.assertFalse(result.threshold_configured)
            self.assertTrue(np.isnan(result.c))
        else:
            self.assertTrue(result.threshold_configured)
            self.assertAlmostEqual(result.c, -result.alpha*(result.sigma_min-result.m_min)-result.drift)

    def test_constraint_rejects_limits_mismatch_and_singular_state(self):
        self.wait(lambda: self.constraint is not None and self.constraint.valid, "initial derivatives")
        for field, value, reason in (("rate_model_valid", False, "servo_rate_model_invalid"),
                                      ("linear_limits_valid", False, "servo_linear_limits_invalid"),
                                      ("rate_model_error", 0.1, "servo_rate_model_mismatch")):
            with self.lock:
                old = getattr(self.__class__, field)
                setattr(self.__class__, field, value)
            self.wait(lambda: not self.constraint.valid and reason in self.constraint.reason, reason)
            with self.lock:
                setattr(self.__class__, field, old)
            self.wait(lambda: self.constraint.valid, "derivative recovery")
        with self.lock:
            self.__class__.q = np.zeros(6)
        self.wait(lambda: not self.constraint.valid and "singular" in self.constraint.reason,
                  "singular derivative rejection")

    def test_status_stop_and_recovery(self):
        for code in (1, 2, 3, 4, 5, 99, 0):
            with self.lock:
                self.__class__.code = code
            self.wait(lambda: self.bridge.moveit_status == code and self.kinematic.moveit_status == code,
                      "status forwarding %d" % code)
            if code in (0, 1):
                self.assertTrue(np.any(values(self.bridge.output)))
            else:
                np.testing.assert_array_equal(values(self.bridge.output), np.zeros(6))

    def test_singular_jacobian_is_reported_not_invented_from_status(self):
        with self.lock:
            self.__class__.q = np.zeros(6)
            self.__class__.code = 2
        self.wait(lambda: self.kinematic.transform_chain_valid and self.kinematic.sigma_min == 0,
                  "singular RobotState metrics")
        self.assertTrue(np.isinf(self.kinematic.condition_number))
        self.assertEqual(self.kinematic.moveit_status, 2)
        np.testing.assert_array_equal(values(self.bridge.output), np.zeros(6))

    def test_wall_watchdogs_with_frozen_ros_clock(self):
        if not rospy.get_param("/use_sim_time", False):
            self.skipTest("run with use_sim_time:=true for the paused-clock check")
        with self.lock:
            self.__class__.pause_clock = True
        try:
            self.wait(lambda: not self.kinematic.transform_chain_valid and
                      self.bridge.reason != "ok" and not np.any(values(self.bridge.output)),
                      "wall watchdogs with frozen /clock")
        finally:
            with self.lock:
                self.__class__.pause_clock = False
        self.wait(lambda: self.kinematic.transform_chain_valid, "clock recovery")

    def test_input_status_tf_and_joint_watchdogs(self):
        for flag, bridge_stops in (("send_input", True), ("send_status", True),
                                   ("send_tf", True), ("send_joints", False)):
            with self.lock:
                setattr(self.__class__, flag, False)
            self.wait(lambda: not self.kinematic.transform_chain_valid, flag + " feedback invalidation")
            if bridge_stops:
                self.wait(lambda: self.bridge.reason != "ok" and not np.any(values(self.bridge.output)),
                          flag + " zero velocity")
            with self.lock:
                setattr(self.__class__, flag, True)
            self.wait(lambda: self.kinematic.transform_chain_valid and self.bridge.reason == "ok", flag + " recovery")

    def test_bad_frame_stamp_joint_coverage_and_model_tf(self):
        for field, value in (("input_frame", "camera"), ("input_age", 1.0),
                             ("input_age", -1.0), ("malformed", True), ("tf_error", 0.2)):
            with self.lock:
                previous = getattr(self.__class__, field)
                setattr(self.__class__, field, value)
            self.wait(lambda: not self.kinematic.transform_chain_valid, field + " invalidation")
            if field in ("input_frame", "input_age"):
                self.wait(lambda: not np.any(values(self.bridge.output)), field + " zero velocity")
            with self.lock:
                setattr(self.__class__, field, previous)
            self.wait(lambda: self.kinematic.transform_chain_valid, field + " recovery")


if __name__ == "__main__":
    rostest.rosrun("omnimap_kinematics", "kinematic_chain", ChainTest)
