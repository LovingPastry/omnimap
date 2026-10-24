#!/usr/bin/env python3
"""Collect read-only source/runtime evidence; a missing master is explicit."""
import argparse
from datetime import datetime
from pathlib import Path
import os
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-source", type=Path, default=Path.home()/"ros/catkin_ws/src")
    parser.add_argument("--servo-ns", default="/servo_server")
    parser.add_argument("--output", type=Path, default=Path.home()/(
        "omnimap_kinematic_inventory_"+datetime.now().strftime("%Y%m%d_%H%M%S")+".txt"))
    args = parser.parse_args()
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
        if master_ok:
            run(["rosparam", "get", args.servo_ns])
            run(["rosnode", "info", args.servo_ns])
            run(["rostopic", "list"])
            for topic in ("/omnimap/kinematic_status", "/omnimap/bridge_status", "/omnimap/servo_twist_trace",
                          "/servo_server/delta_twist_camera", "/servo_server/delta_twist_cmds",
                          "/servo_server/status", "/joint_states"):
                run(["rostopic", "info", topic])
                run(["rostopic", "echo", "-n", "1", topic], timeout=3)
        else:
            output.write("RUNTIME_UNVERIFIED: ROS master unavailable or ROS environment not sourced.\n")
    print(args.output)
    return 0 if master_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
