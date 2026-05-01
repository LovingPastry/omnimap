#!/usr/bin/env bash

# Lightweight execution-side environment for info_flow_servo_runtime.py
# Assumes ROS Noetic and robot-side stack are already installed.

if [ -n "${VIRTUAL_ENV:-}" ]; then
    # keep user-selected venv
    :
elif [ -n "${SERVO_VENV_PATH:-}" ] && [ -f "${SERVO_VENV_PATH}/bin/activate" ]; then
    # shellcheck disable=SC1090
    source "${SERVO_VENV_PATH}/bin/activate"
fi

source /opt/ros/noetic/setup.bash

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "${REPO_ROOT}/ros_ws/devel/setup.bash" ]; then
    source "${REPO_ROOT}/ros_ws/devel/setup.bash"
fi

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"
export ROS_MASTER_URI="${ROS_MASTER_URI:-http://127.0.0.1:11311}"

if [ -n "${ROS_IP:-}" ]; then
    export ROS_IP
elif [ -n "${ROS_HOSTNAME:-}" ]; then
    export ROS_HOSTNAME
fi
