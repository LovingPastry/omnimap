#!/usr/bin/env python3
"""Base-expressed camera -> tool reference-point velocity, with a wall watchdog.

No Jacobian is computed here. Every publication has a BridgeStatus companion;
singularity metrics come exclusively from the MoveIt RobotState monitor.
"""
import copy
import math
import threading
import time

import rospy
import tf2_ros
from geometry_msgs.msg import TwistStamped
from omnimap_msgs.msg import BridgeStatus
from std_msgs.msg import Int8


STATUS_TEXT = {0: "NO_WARNING", 1: "DECELERATE_FOR_SINGULARITY",
               2: "HALT_FOR_SINGULARITY", 3: "DECELERATE_FOR_COLLISION",
               4: "HALT_FOR_COLLISION", 5: "JOINT_BOUND"}


def components(msg):
    v, w = msg.twist.linear, msg.twist.angular
    return (v.x, v.y, v.z, w.x, w.y, w.z)


def tool_velocity(values, offset):
    vx, vy, vz, wx, wy, wz = values
    x, y, z = offset
    return (vx - (wy*z - wz*y), vy - (wz*x - wx*z),
            vz - (wx*y - wy*x), wx, wy, wz)


class CameraTwistBridge:
    def __init__(self):
        self.base = rospy.get_param("~base_frame", "base_link").lstrip("/")
        self.camera = rospy.get_param("~camera_frame", "cam_1_color_optical_frame").lstrip("/")
        self.tool = rospy.get_param("~tool_frame", "tool0").lstrip("/")
        if not all((self.base, self.camera, self.tool)):
            raise ValueError("bridge frames must be nonempty")
        self.hz = self.positive_param("publish_hz", 50.0)
        self.timeout = self.positive_param("stale_timeout", 0.3)
        self.status_timeout = self.positive_param("status_timeout", 0.3)
        self.filter_mode = rospy.get_param("~filter_mode", "none")
        if self.filter_mode not in ("none", "kalman"):
            raise ValueError("filter_mode must be none or kalman")
        self.q = [self.positive_param("process_noise_linear", 1e-4)]*3 + [
            self.positive_param("process_noise_angular", 1e-4)]*3
        self.r = [self.positive_param("measurement_noise_linear", 1e-3)]*3 + [
            self.positive_param("measurement_noise_angular", 1e-3)]*3
        self.filtered, self.variance, self.filtered_generation = None, None, -1
        self.lock = threading.RLock()
        self.input, self.input_wall, self.generation = None, 0.0, 0
        self.status, self.status_wall, self.status_stamp = -1, 0.0, rospy.Time()
        self.buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.buffer)
        input_topic = rospy.get_param("~input_topic", "/servo_server/delta_twist_camera")
        output_topic = rospy.get_param("~output_topic", "/servo_server/delta_twist_cmds")
        if rospy.resolve_name(input_topic) == rospy.resolve_name(output_topic):
            raise ValueError("bridge input and output topics must differ")
        self.input_topic, self.output_topic = rospy.resolve_name(input_topic), rospy.resolve_name(output_topic)
        self.pub = rospy.Publisher(output_topic, TwistStamped, queue_size=1)
        self.diag = rospy.Publisher(rospy.get_param("~bridge_status_topic", "/omnimap/bridge_status"),
                                    BridgeStatus, queue_size=10)
        self.sub = rospy.Subscriber(input_topic, TwistStamped, self.input_cb, queue_size=1)
        self.status_topic = self.effective_status_topic()
        self.status_sub = rospy.Subscriber(self.status_topic, Int8, self.status_cb, queue_size=1)
        self.closed = threading.Event()
        rospy.on_shutdown(self.shutdown)
        # rospy.Timer stops under paused /clock. The watchdog must not.
        self.worker = threading.Thread(target=self.run, daemon=True)
        self.worker.start()
        rospy.loginfo("bridge: %s -> %s B=%s C=%s T=%s filter=%s status_topic=%s",
                      input_topic, output_topic, self.base, self.camera, self.tool, self.filter_mode,
                      self.status_topic)

    @staticmethod
    def effective_status_topic():
        explicit = rospy.get_param("~status_topic", "")
        if explicit:
            return rospy.resolve_name(explicit)
        namespace = rospy.resolve_name(rospy.get_param("~servo_ns", "/servo_server")).rstrip("/")
        topic = rospy.get_param(namespace + "/status_topic", namespace + "/status")
        if not isinstance(topic, str) or not topic or topic.startswith("~"):
            raise ValueError("Servo status_topic must be a nonempty global or relative ROS name")
        # Match roscpp's private NodeHandle(servo_ns).resolveName(). Relative
        # names in Servo's config belong under Servo, not the bridge node.
        return rospy.resolve_name(topic if topic.startswith("/") else namespace + "/" + topic)

    @staticmethod
    def positive_param(name, default):
        value = float(rospy.get_param("~" + name, default))
        if not math.isfinite(value) or value <= 0:
            raise ValueError(name + " must be finite and positive")
        return value

    def input_cb(self, msg):
        with self.lock:
            # Invalid input replaces the previous command; never hold old motion.
            self.input, self.input_wall = msg, time.monotonic()
            self.generation += 1

    def status_cb(self, msg):
        with self.lock:
            self.status, self.status_wall = msg.data, time.monotonic()
            self.status_stamp = rospy.Time.now()

    def build(self):
        """Called under lock. Failure always returns an explicit zero output."""
        now, wall = rospy.Time.now(), time.monotonic()
        record = BridgeStatus()
        record.header.stamp = now
        record.header.frame_id = self.base
        record.camera_frame, record.task_frame = self.camera, self.tool
        record.input_topic, record.output_topic = self.input_topic, self.output_topic
        record.input = self.input if self.input is not None else TwistStamped()
        record.output.header.stamp = now
        record.output.header.frame_id = self.base
        record.moveit_status, record.status_received_stamp = self.status, self.status_stamp
        record.camera_tf_age = record.tool_tf_age = math.nan
        try:
            if self.input is None:
                raise ValueError("input_missing")
            age = (now - self.input.header.stamp).to_sec()
            if self.input.header.stamp.to_sec() <= 0 or age < -0.05:
                raise ValueError("input_invalid_stamp")
            if age > self.timeout or wall - self.input_wall > self.timeout:
                raise ValueError("input_stale")
            if self.input.header.frame_id.lstrip("/") != self.base:
                raise ValueError("input_frame_mismatch")
            values = components(self.input)
            if not all(math.isfinite(v) for v in values):
                raise ValueError("input_nonfinite")
            status_age = (now - self.status_stamp).to_sec()
            if (self.status_stamp.to_sec() <= 0 or status_age < -0.05 or
                    status_age > self.status_timeout or wall-self.status_wall > self.status_timeout):
                raise ValueError("status_stale_or_missing")
            # Both transforms at the input sample time, never two independent
            # Time(0) snapshots. tf2 rejects unavailable/expired dynamic data.
            transforms = [self.buffer.lookup_transform(
                self.base, frame, self.input.header.stamp, rospy.Duration(0)
            ) for frame in (self.camera, self.tool)]
            # Nonblocking: tf2's Python timeout polls ROS time and could hang
            # this wall watchdog indefinitely when /clock is paused.
            ages = [(now-t.header.stamp).to_sec() if t.header.stamp.to_sec() else 0.0
                    for t in transforms]  # zero stamp is a static transform
            record.camera_tf_age, record.tool_tf_age = ages
            if any(a < -0.05 or a > self.timeout for a in ages):
                raise ValueError("tf_stale")
            c, t = [tr.transform.translation for tr in transforms]
            offset = (c.x-t.x, c.y-t.y, c.z-t.z)
            result = tool_velocity(values, offset)
            if not all(math.isfinite(v) for v in result):
                raise ValueError("tf_nonfinite")
            record.transform_chain_valid = True
            if self.status not in (0, 1):
                raise ValueError("moveit_" + STATUS_TEXT.get(self.status, "UNKNOWN"))
            # A zero command is a stop, not a noisy measurement to smooth.
            if not any(values):
                self.filtered = self.variance = None
                result = (0.0,)*6
            elif self.filter_mode == "kalman":
                if self.filtered is None:
                    self.filtered, self.variance = list(result), list(self.r)
                elif self.filtered_generation != self.generation:
                    for i, measurement in enumerate(result):
                        p = self.variance[i] + self.q[i]
                        gain = p / (p + self.r[i])
                        self.filtered[i] += gain*(measurement-self.filtered[i])
                        self.variance[i] = (1-gain)*p
                result = self.filtered
            self.filtered_generation = self.generation
            v, w = record.output.twist.linear, record.output.twist.angular
            v.x, v.y, v.z, w.x, w.y, w.z = result
            record.reason = "ok"
        except (ValueError, tf2_ros.TransformException, rospy.ROSException) as exc:
            self.filtered = self.variance = None
            record.reason = str(exc)
        return record

    def publish(self):
        with self.lock:
            if self.closed.is_set():
                return
            record = self.build()
            # rospy assigns Header.seq during serialization. Publish a copy so
            # the companion record retains only source identity, not a new seq.
            published = copy.deepcopy(record.output)
            self.pub.publish(published)
            record.output.header.seq = published.header.seq
            self.diag.publish(record)
            rospy.loginfo("bridge_sample: input_stamp=%.9f input_seq=%d output_stamp=%.9f output_seq=%d "
                          "input_frame=%s output_frame=%s tf_age=[%.6g %.6g] "
                          "input=%s output=%s status=%d status_stamp=%.9f valid=%s reason=%s",
                          record.input.header.stamp.to_sec(), record.input.header.seq,
                          record.output.header.stamp.to_sec(), record.output.header.seq,
                          record.input.header.frame_id, record.output.header.frame_id,
                          record.camera_tf_age, record.tool_tf_age, components(record.input),
                          components(record.output), record.moveit_status,
                          record.status_received_stamp.to_sec(), record.transform_chain_valid, record.reason)

    def run(self):
        while not self.closed.is_set() and not rospy.is_shutdown():
            try:
                self.publish()
            except Exception as exc:
                rospy.logerr("bridge watchdog exception: %s", exc)
                zero = TwistStamped()
                zero.header.stamp, zero.header.frame_id = rospy.Time.now(), self.base
                self.pub.publish(zero)
            self.closed.wait(1.0/self.hz)

    def shutdown(self):
        self.closed.set()
        # Serialize with any in-flight publication; final output must be zero.
        with self.lock:
            zero = TwistStamped()
            zero.header.stamp, zero.header.frame_id = rospy.Time.now(), self.base
            self.pub.publish(zero)


if __name__ == "__main__":
    rospy.init_node("omnimap_camera_twist_bridge")
    CameraTwistBridge()
    rospy.spin()
