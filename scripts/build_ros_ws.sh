#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROS_WS="$REPO_ROOT/ros_ws"
SYSTEM_EMPY="/usr/lib/python3/dist-packages/em.py"
BUILD_PACKAGES="omnimap_msgs"
if [ "$#" -gt 1 ]; then
  echo "Usage: $0 [--messages-only|--with-kinematics]" >&2
  exit 2
fi
case "${1:-}" in
  ""|--messages-only) ;;
  --with-kinematics) BUILD_PACKAGES="omnimap_msgs;omnimap_kinematics" ;;
  *) echo "Usage: $0 [--messages-only|--with-kinematics]" >&2; exit 2 ;;
esac

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
      '-DCATKIN_WHITELIST_PACKAGES=$BUILD_PACKAGES' \
      -DPYTHON_EXECUTABLE=/usr/bin/python3 \
      -DEMPY_SCRIPT=$SYSTEM_EMPY \
      -DPY_EM=$SYSTEM_EMPY
  "
