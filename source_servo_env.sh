#!/usr/bin/env bash

# Lightweight execution-side environment for info_flow_servo_runtime.py
# Assumes ROS Noetic and robot-side stack are already installed.

_activate_servo_conda() {
    local target="${1:-}"
    if [ -z "$target" ]; then
        return 1
    fi

    if ! command -v conda >/dev/null 2>&1; then
        return 1
    fi

    local conda_base
    conda_base="$(conda info --base 2>/dev/null || true)"
    if [ -z "$conda_base" ] || [ ! -f "$conda_base/etc/profile.d/conda.sh" ]; then
        return 1
    fi

    # shellcheck disable=SC1090
    source "$conda_base/etc/profile.d/conda.sh"
    conda activate "$target"
}

if [ -n "${CONDA_PREFIX:-}" ]; then
    # keep user-selected conda env
    :
elif [ -n "${SERVO_CONDA_ENV:-}" ]; then
    _activate_servo_conda "${SERVO_CONDA_ENV}" || echo "[source_servo_env] WARN: failed to activate SERVO_CONDA_ENV=${SERVO_CONDA_ENV}"
elif [ -n "${VIRTUAL_ENV:-}" ]; then
    # keep user-selected venv
    :
elif [ -n "${SERVO_VENV_PATH:-}" ] && [ -f "${SERVO_VENV_PATH}/bin/activate" ]; then
    # legacy fallback for venv users
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
