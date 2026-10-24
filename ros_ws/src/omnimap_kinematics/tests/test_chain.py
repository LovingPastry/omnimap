#!/usr/bin/env python3
"""Real ROS nodes + MoveIt RobotState, synthetic chain, no hardware topics.

Independent NumPy forward kinematics and finite differences check the monitor's
frame, reference point, variable order, normalization and measured speed.
"""
import threading
import time
import unittest
import xml.etree.ElementTree as ET

import numpy as np
import rospy
import rostest
import tf2_ros
from geometry_msgs.msg import TransformStamped, TwistStamped
from omnimap_msgs.msg import BridgeStatus, KinematicStatus, ServoTwistTrace
from sensor_msgs.msg import JointState
from std_msgs.msg import Int8
from rosgraph_msgs.msg import Clock
from tf.transformations import euler_matrix, rotation_matrix, quaternion_from_matrix


def values(msg):
    v, w = msg.twist.linear, msg.twist.angular
    return np.array([v.x, v.y, v.z, w.x, w.y, w.z])


class ChainTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rospy.init_node("test_kinematic_chain")
        cls.lock = threading.RLock()
        cls.kinematic, cls.bridge, cls.output = None, None, None
        cls.subs = [rospy.Subscriber(topic, kind, lambda msg, key=key: setattr(cls, key, msg))
                    for topic, kind, key in [("/test/kinematic", KinematicStatus, "kinematic"),
                        ("/test/bridge", BridgeStatus, "bridge"), ("/test/tool_twist", TwistStamped, "output")]]
        cls.joint_pub = rospy.Publisher("/test/joints", JointState, queue_size=1)
        cls.status_pub = rospy.Publisher("/test/status", Int8, queue_size=1)
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
            self.__class__.input_frame = "base_link"
            self.__class__.malformed = False
            self.__class__.tf_error = 0.0
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
                cls.clock_pub.publish(Clock(rospy.Time.from_sec(time.time())))
                if not hasattr(cls, "q"):
                    continue
                now = rospy.Time.now()
                if now.is_zero():
                    continue
                # Publish TF ahead of the input, simulating sensor/compute delay.
                poses = cls.fk(cls.q)
                if cls.send_tf:
                    transforms = []
                    for frame in ("tool0", "cam_1_color_optical_frame"):
                        tr = TransformStamped()
                        tr.header.stamp, tr.header.frame_id, tr.child_frame_id = now, "base_link", frame
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
                    js.header.stamp = stamp
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
                    cls.input_pub.publish(msg)
                    # rospy has now assigned msg.header.seq. Nested headers in
                    # the trace preserve that actual wire sequence.
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
