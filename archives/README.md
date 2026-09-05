# Archives Directory

This directory contains deprecated, backup, and demo files that are not part of the active codebase.

## Structure

### `deprecated_nodes/`
Obsolete ROS node implementations replaced by the `info_flow/` architecture:
- `omni_ros_node.py`: Legacy single-node ROS interface (superseded by `info_flow_node.py`)
- `fisher_field.py`: Standalone Fisher field computation (now integrated in `info_flow_planning_node.py`)

**Note**: These files are kept for reference only. The current system uses the three-loop architecture in `info_flow/`.

### `configs_backup/`
Runtime-generated configuration snapshots:
- `RenderOption_2026-04-06-15-21-41.json`: Temporary rendering options from a specific run

### `demo_data/`
Reserved for demonstration data and videos (currently empty).

---

**Created**: 2026-09-05  
**Purpose**: Code space cleanup to improve repository organization
