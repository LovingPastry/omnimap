#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROS_WS="$REPO_ROOT/ros_ws"
SYSTEM_EMPY="/usr/lib/python3/dist-packages/em.py"

env \
  -u PYTHONPATH \
  -u PYTHONHOME \
  -u CONDA_PREFIX \
  -u CONDA_DEFAULT_ENV \
  -u CONDA_PROMPT_MODIFIER \
  -u _CE_CONDA \
  -u _CE_M \
  -u VIRTUAL_ENV \
  PATH=/usr/bin:/bin:/usr/sbin:/sbin \
  bash --noprofile --norc -lc "
    source /opt/ros/noetic/setup.bash
    cd \"$ROS_WS\"
    catkin_make \
      -DPYTHON_EXECUTABLE=/usr/bin/python3 \
      -DEMPY_SCRIPT=$SYSTEM_EMPY \
      -DPY_EM=$SYSTEM_EMPY
  "
