#!/usr/bin/env python3
"""Real Servo timers, RPC, monitor and bridge on synthetic /test/* inputs."""
import copy
import os
import sys
import threading
import time
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import rospy
import rostest
import tf2_ros
from geometry_msgs.msg import TransformStamped, TwistStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Int8
from omnimap_msgs.msg import SphericalCommand, ServoTwistTrace
from tf.transformations import quaternion_from_matrix
from test_chain import ChainTest, values

# Resolve the checkout through this package, also when linked into catkin_ws.
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT/"info_flow"))
from info_flow_servo_runtime import InfoFlowServoRuntime


class ServoPreviewChain(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        args = NS(world_frame="base_link", camera_frame="cam_1_color_optical_frame", cmd_frame="base_link",
            cmd_topic="/test/camera_twist", twist_trace_topic="/test/trace",
            spherical_cmd_topic="/test/final", spherical_cmd_raw_topic="/test/raw",
            kinematic_joint_topic="/test/joints", kinematic_status_topic="/test/kinematic",
            kinematic_preview_service="/test_monitor/evaluate_model", kinematic_preview_mode="observe",
            kinematic_preview_source=os.environ.get("OMNIMAP_TEST_PREVIEW_SOURCE", "query"),
            kinematic_model_topic="/test/model",
            kinematic_preview_hz=10., kinematic_preview_timeout_sec=.1,
            kinematic_cycle_projection_budget_sec=.02, kinematic_projection_recovery_sec=.1,
            servo_hz=10., pose_stale_timeout_sec=.4, spherical_cmd_timeout_sec=.6,
            adaptive_cmd_timeout_scale=1.5, adaptive_cmd_timeout_cap_sec=.6,
            linear_vel_max=.1, linear_deadband=0., angular_speed_max=1., enable_angular=False,
            radial_gain=0., radial_i_gain=0., radial_deadband=.001, radial_integral_limit=.05,
            radial_vel_max=.015, angular_gain=1., angular_deadband=.001,
            linear_accel_max=0., angular_accel_max=0., status_log_interval_sec=10.)
        cls.executing = os.environ.get("OMNIMAP_TEST_EXECUTION") == "1"
        if cls.executing:
            args.kinematic_preview_mode, args.kinematic_preview_source = "enforce", "stream"
            args.kinematic_execution_max_age, args.kinematic_execution_joint_delta = .2, .05
            args.kinematic_execution_residual, args.kinematic_execution_bridge_node = .01, "/test_bridge"
            args.linear_vel_max = .002
        cls.runtime = InfoFlowServoRuntime(args)
        cls.lock = threading.RLock()
        cls.code, cls.send_joints, cls.policy_stop = 0, True, False
        cls.raw_pub = rospy.Publisher("/test/raw", SphericalCommand, queue_size=1)
        cls.final_pub = rospy.Publisher("/test/final", SphericalCommand, queue_size=1)
        cls.joint_pub = rospy.Publisher("/test/joints", JointState, queue_size=1)
        cls.status_pub = rospy.Publisher("/test/servo/feedback", Int8, queue_size=1)
        cls.twists, cls.traces = [], []
        cls.tool_velocity, cls.tool_wall = np.zeros(6), 0.
        cls.twist_sub = rospy.Subscriber("/test/camera_twist", TwistStamped, cls.twists.append, queue_size=100)
        cls.trace_sub = rospy.Subscriber("/test/trace", ServoTwistTrace, cls.traces.append, queue_size=100)
        cls.tool_sub = rospy.Subscriber("/test/tool_twist", TwistStamped, cls.tool_cb, queue_size=1)
        cls.tf = tf2_ros.TransformBroadcaster()
        ChainTest.joints = ET.fromstring(rospy.get_param("/robot_description")).findall("joint")
        cls.q = np.array([.2, -.8, 1.3, .4, -.7, .5])
        cls.poses = ChainTest.fk(cls.q)
        cls.closed = threading.Event()
        cls.worker = threading.Thread(target=cls.feed, daemon=True)
        cls.worker.start()

    @classmethod
    def tearDownClass(cls):
        cls.closed.set()
        cls.worker.join(timeout=2)
        if cls.runtime.preview_timer is not None:
            cls.runtime.preview_timer.shutdown()
        cls.runtime.timer.shutdown()

    @classmethod
    def tool_cb(cls, msg):
        with cls.lock:
            cls.tool_velocity, cls.tool_wall = values(msg), time.monotonic()

    @classmethod
    def feed(cls):
        next_command, tick, previous = 0., 0, time.monotonic()
        while not cls.closed.wait(.02) and not rospy.is_shutdown():
            now = rospy.Time.now()
            with cls.lock:
                q = cls.q.copy()
                dq = np.zeros(6)
                wall = time.monotonic()
                if cls.executing and wall-cls.tool_wall < .25:
                    dq = np.linalg.solve(ChainTest.independent_jacobian(ChainTest, q), cls.tool_velocity)
                    q += dq*min(wall-previous, .05)
                    cls.q = q.copy()
                previous = wall
            poses = ChainTest.fk(q)
            transforms = []
            for frame in ("tool0", "cam_1_color_optical_frame"):
                pose = poses[frame]
                tr = TransformStamped()
                tr.header.stamp, tr.header.frame_id, tr.child_frame_id = now, "base_link", frame
                tr.transform.translation.x, tr.transform.translation.y, tr.transform.translation.z = pose[:3, 3]
                quat = quaternion_from_matrix(pose)
                (tr.transform.rotation.x, tr.transform.rotation.y,
                 tr.transform.rotation.z, tr.transform.rotation.w) = quat
                transforms.append(tr)
            cls.tf.sendTransform(transforms)
            if cls.send_joints:
                js = JointState(name=["j%d" % i for i in range(6, 0, -1)],
                                position=q[::-1].tolist(), velocity=dq[::-1].tolist())
                js.header.stamp = now
                cls.joint_pub.publish(js)
            cls.status_pub.publish(Int8(cls.code))
            if time.monotonic() >= next_command:
                tick += 1
                next_command = time.monotonic()+.25
                center = poses["cam_1_color_optical_frame"][:3, 3]+[.5, 0, 0]
                raw = SphericalCommand(planner_tick=tick, model_version=7, theta_rate=.01,
                    reference_radius=.5, reference_scene_center=center.tolist(), should_stop=cls.policy_stop)
                raw.header.stamp, raw.header.frame_id = now, "base_link"
                final = copy.deepcopy(raw)
                if cls.executing:
                    final.stop_reason = "synthetic_policy_stop" if final.should_stop else ""
                else:
                    final.should_stop, final.stop_reason = True, "synthetic_fixture_stop"
                    final.theta_rate = final.phi_rate = 0.
                cls.raw_pub.publish(raw)
                cls.final_pub.publish(final)

    def wait(self, predicate, label, timeout=8):
        deadline = time.monotonic()+timeout
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(.02)
        decision = self.runtime.projection_readiness.decision(
            ros_now=rospy.Time.now().to_sec(), wall_now=time.monotonic())
        self.fail("timeout %s; health=%s evaluation=%s proposal=%s" %
            (label, decision, self.runtime.latest_preview_evaluation, self.runtime.latest_cycle_proposal))

    def current_record(self):
        return (self.runtime.latest_stream_projection if self.runtime.preview_source == "stream"
                else self.runtime.latest_preview_evaluation)

    def ready(self):
        proposal = self.runtime.latest_cycle_proposal
        return proposal is not None and proposal.valid

    def wait_record(self, label="consistent model/proposal"):
        captured = []
        def capture():
            proposal, record = self.runtime.latest_cycle_proposal, self.current_record()
            if (proposal is not None and proposal.valid and record is not None
                    and proposal.source_tick == record.constraint.planner_tick):
                captured[:] = [(record, proposal)]
                return True
            return False
        self.wait(capture, label)
        return captured[0]

    def assert_stopped(self):
        self.assertTrue(self.twists)
        for msg in self.twists:
            v, w = msg.twist.linear, msg.twist.angular
            np.testing.assert_array_equal([v.x, v.y, v.z, w.x, w.y, w.z], np.zeros(6))

    def setUp(self):
        self.__class__.code, self.__class__.send_joints, self.__class__.policy_stop = 0, True, False
        self.record, self.proposal = self.wait_record("model and actual-cycle proposal")

    def test_real_timer_rpc_and_projection_keep_final_stop(self):
        record = self.record
        self.assertIsNotNone(record)
        self.assertTrue(record.constraint.reason.startswith("advisory_preview:"))
        self.assertGreater(record.constraint.joint_stamp.to_sec(), 0.)
        self.assertEqual(self.proposal.reason, "advisory_projection")
        self.assert_stopped()

    def test_moveit_halt_clears_model_and_requires_fresh_recovery(self):
        old = self.record
        self.__class__.code = 4
        self.wait(lambda: self.current_record() is None
                  and not self.runtime.projection_readiness.decision(
                      ros_now=rospy.Time.now().to_sec(), wall_now=time.monotonic()).ready, "halt")
        self.assertIsNone(self.runtime.latest_cycle_proposal)
        self.__class__.code = 0
        recovered, _ = self.wait_record("health recovery and new model")
        self.assertIsNot(recovered, old)
        self.assert_stopped()

    def test_joint_stream_loss_and_recovery(self):
        self.__class__.send_joints = False
        self.wait(lambda: not self.runtime.projection_readiness.decision(
            ros_now=rospy.Time.now().to_sec(), wall_now=time.monotonic()).ready, "joint loss")
        self.assertIsNone(self.runtime.latest_cycle_proposal)
        self.__class__.send_joints = True
        self.wait_record("new joints and new model")
        self.assert_stopped()

    def test_policy_stop_prevents_preview(self):
        self.__class__.policy_stop = True
        self.wait(lambda: self.runtime.latest_raw_cmd is not None and self.runtime.latest_raw_cmd.msg.should_stop
                  and self.current_record() is None
                  and self.runtime.latest_cycle_proposal is None, "policy stop")
        self.assert_stopped()


class StreamServoChain(ServoPreviewChain):
    def test_moving_samples_get_new_canonical_projection_without_rpc(self):
        self.assertIsNone(self.runtime.preview_query)
        self.assertIsNone(self.runtime.preview_timer)
        records = []
        for step in range(1, 6):
            target = .2+.005*step
            with self.lock:
                self.__class__.q[0] = target
            captured = []
            def at_target():
                record, proposal = self.current_record(), self.runtime.latest_cycle_proposal
                if (record is not None and proposal is not None and proposal.valid
                        and proposal.source_tick == record.constraint.planner_tick
                        and abs(record.model.joint_position[0]-target) < 1e-10):
                    captured[:] = [record]
                    return True
                return False
            self.wait(at_target, "moving canonical projection")
            records.append(captured[0])
        self.assertGreater(records[-1].model.model_sequence, records[0].model.model_sequence)
        self.assertGreater(np.linalg.norm(np.asarray(records[-1].constraint.joint_rate_matrix)-
                                          records[0].constraint.joint_rate_matrix), 1e-5)
        self.assert_stopped()


class ServoExecutionChain(unittest.TestCase):
    setUpClass = classmethod(ServoPreviewChain.setUpClass.__func__)
    tearDownClass = classmethod(ServoPreviewChain.tearDownClass.__func__)
    feed = classmethod(ServoPreviewChain.feed.__func__)
    tool_cb = classmethod(ServoPreviewChain.tool_cb.__func__)
    wait = ServoPreviewChain.wait
    current_record = ServoPreviewChain.current_record

    def moving(self):
        return bool(self.twists) and np.linalg.norm(values(self.twists[-1])) > 1e-6

    def stopped(self):
        return bool(self.twists) and not np.any(values(self.twists[-1]))

    def setUp(self):
        self.__class__.code, self.__class__.send_joints, self.__class__.policy_stop = 0, True, False
        self.wait(self.moving, "projected nonzero output", timeout=12)

    def test_nonzero_projected_output_drives_ideal_joint_model(self):
        with self.lock:
            start = self.q.copy()
        count = len(self.traces)
        self.wait(lambda: len(self.traces) >= count+5, "five execution cycles")
        motion = [trace for trace in self.traces[count:] if trace.reason == "motion"
                  and np.linalg.norm(values(trace.command)) > 1e-6]
        self.assertTrue(motion)
        self.assertTrue(any(0 < abs(trace.spherical_rates[0]) < .009 for trace in motion))
        self.assertTrue(all(np.linalg.norm(values(trace.command)[:3]) <= .002+1e-9 for trace in motion))
        with self.lock:
            self.assertGreater(np.linalg.norm(self.q-start), 1e-5)

    def test_moveit_halt_zero_and_fresh_model_recovery(self):
        self.__class__.code = 4
        self.wait(lambda: self.stopped() and not self.runtime.projection_readiness.decision(
            ros_now=rospy.Time.now().to_sec(), wall_now=time.monotonic()).ready, "halt to zero")
        begin = len(self.twists)
        time.sleep(.25)
        self.assertTrue(all(not np.any(values(msg)) for msg in self.twists[begin:]))
        self.__class__.code = 0
        self.wait(self.moving, "fresh health/model nonzero recovery")

    def test_policy_stop_and_joint_loss_zero_with_recovery(self):
        self.__class__.policy_stop = True
        self.wait(lambda: self.runtime.latest_cmd.msg.should_stop and self.stopped(), "policy zero")
        self.__class__.policy_stop = False
        self.wait(self.moving, "policy recovery")
        self.__class__.send_joints = False
        self.wait(lambda: self.stopped() and not self.runtime.projection_readiness.decision(
            ros_now=rospy.Time.now().to_sec(), wall_now=time.monotonic()).ready, "joint loss zero")
        begin = len(self.twists)
        time.sleep(.2)
        self.assertTrue(all(not np.any(values(msg)) for msg in self.twists[begin:]))
        self.__class__.send_joints = True
        self.wait(self.moving, "joint stream recovery")


if __name__ == "__main__":
    streaming = os.environ.get("OMNIMAP_TEST_PREVIEW_SOURCE") == "stream"
    executing = os.environ.get("OMNIMAP_TEST_EXECUTION") == "1"
    rostest.rosrun("omnimap_kinematics", "servo_execution_chain" if executing else
                  ("servo_stream_chain" if streaming else "servo_preview_chain"),
                  ServoExecutionChain if executing else (StreamServoChain if streaming else ServoPreviewChain))
