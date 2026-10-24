#!/usr/bin/env python3
"""Collect read-only source/runtime evidence; a missing master is explicit."""
import argparse
from datetime import datetime
from pathlib import Path
import os
import subprocess
import json
import socket
from xmlrpc.client import ServerProxy, Error as XmlRpcError


def servo_topic(namespace, topic):
    if not isinstance(topic, str) or not topic or topic.startswith("~"):
        raise ValueError("invalid Servo topic: " + repr(topic))
    return topic if topic.startswith("/") else namespace.rstrip("/") + "/" + topic


def servo_config(namespace):
    previous_timeout = socket.getdefaulttimeout()
    socket.setdefaulttimeout(3)
    try:
        with ServerProxy(os.environ.get("ROS_MASTER_URI", "http://localhost:11311")) as master:
            code, message, config = master.getParam("/omnimap_kinematic_inventory", namespace)
        if code != 1 or not isinstance(config, dict):
            raise ValueError("Servo configuration unavailable: " + str(message))
        return config
    finally:
        socket.setdefaulttimeout(previous_timeout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-source", type=Path, default=Path.home()/"ros/catkin_ws/src")
    parser.add_argument("--servo-ns", default="/servo_server")
    parser.add_argument("--input-topic", default="/servo_server/delta_twist_camera")
    parser.add_argument("--output", type=Path, default=Path.home()/(
        "omnimap_kinematic_inventory_"+datetime.now().strftime("%Y%m%d_%H%M%S")+".txt"))
    args = parser.parse_args()
    args.servo_ns = "/" + args.servo_ns.strip("/")
    repo = Path(__file__).resolve().parents[1]
    # Exclusive creation preserves previous captures.
    with args.output.open("x") as output:
        def run(command, timeout=5):
            output.write("\n$ " + " ".join(str(item) for item in command) + "\n")
            try:
                result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                        text=True, timeout=timeout)
                output.write(result.stdout)
                output.write("exit_code="+str(result.returncode)+"\n")
                return result.returncode
            except (OSError, subprocess.TimeoutExpired) as exc:
                output.write(str(exc)+"\n")
                return 2

        output.write("captured="+datetime.now().astimezone().isoformat()+"\n")
        for name in ("ROS_MASTER_URI", "ROS_IP", "ROS_HOSTNAME"):
            output.write(name+"="+os.environ.get(name, "<unset>")+"\n")
        run(["git", "-C", repo, "rev-parse", "HEAD"])
        run(["rg", "-n", "delta_twist|base_link|tool0|cam_1|runpy|filter_mode|kinematic",
             args.robot_source/"robot_else_nodes/scripts/camera_twist_bridge.py",
             args.robot_source/"robot_else_nodes/scripts/camera_twist_bridge_kf.py",
             args.robot_source/"robot_else_nodes/launch/launch_robot.launch",
             args.robot_source/"universal_robot/ur5_moveit_config/config/ur5_servo.yaml",
             repo/"ros_ws/src/omnimap_kinematics/scripts/camera_twist_bridge.py",
             repo/"ros_ws/src/omnimap_kinematics/src/omnimap_kinematic_monitor.cpp"])
        master_ok = run(["rosnode", "list"]) == 0
        config_ok = False
        if master_ok:
            run(["rosparam", "get", args.servo_ns])
            run(["rosnode", "info", args.servo_ns])
            run(["rostopic", "list"])
            topics = ["/omnimap/kinematic_status", "/omnimap/bridge_status", "/omnimap/servo_twist_trace",
                      args.input_topic]
            try:
                config = servo_config(args.servo_ns)
                configured_topics = [servo_topic(args.servo_ns, config[key]) for key in
                                     ("cartesian_command_in_topic", "status_topic", "joint_topic")]
                output.write("effective_servo_config=" + json.dumps(config, sort_keys=True) + "\n")
                topics.extend(configured_topics)
                config_ok = True
            except (OSError, ValueError, KeyError, XmlRpcError) as exc:
                output.write("CONFIG_UNVERIFIED: " + str(exc) + "\n")
            for topic in dict.fromkeys(topics):
                run(["rostopic", "info", topic])
                run(["rostopic", "echo", "-n", "1", topic], timeout=3)
        else:
            output.write("RUNTIME_UNVERIFIED: ROS master unavailable or ROS environment not sourced.\n")
    print(args.output)
    return 0 if master_ok and config_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
