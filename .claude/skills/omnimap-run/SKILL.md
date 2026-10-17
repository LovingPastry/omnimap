---
name: omnimap-run
description: Run OmniMap's entry points — offline incremental mapping on Replica/ScanNet, mesh generation, the Fisher-guided simulation closed loop, and the online ROS three-loop node — plus the unified logging flags and dataset path setup. Use when launching, configuring, or debugging any of these runs.
---

# Running OmniMap

## Offline incremental mapping

```bash
python demo.py --dataset replica --scene room_0 [--vis_gui] [--start N] [--length M]
python demo.py --dataset scannet --scene scene0000_00
```

Generate a mesh from rendered outputs:

```bash
python tools/tsdf_integrate.py --dataset replica --scene room_0
```

## Simulation closed loop (Fisher information-guided)

```bash
python sim/main.py \
  --pcd_path <path_to_ply> \
  --save_dir sim/sim_outputs/test \
  --fisher_step_scale 1e-4 \
  --radial_gain 0.2 \
  --angular_gain 2.0 \
  [--vis_gui]
```

## Online ROS node (three-loop: tracking + planning + servo)

```bash
python info_flow/info_flow_node.py --config config/rtabmap_config.yaml
```

## Logging control

Entry scripts (`demo.py`, `sim/main.py`, `info_flow/*.py`) share a unified logging interface:

- Log sections: `main`, `tsdf`, `gaussian`, `fisher`, `planner`, `profile`, `all`
- Log levels: `DEBUG`, `INFO`, `WARNING`, `ERROR`
- Log profiles: `quiet`, `default`, `debug`

`--log_section` is repeatable; each occurrence adds one section.

```bash
# Only main + planner sections at INFO level
python info_flow/info_flow_node.py \
  --config config/rtabmap_config.yaml \
  --log_section main --log_section planner \
  --log_min_level INFO

# Fisher + planner debug
python sim/main.py --pcd_path <path> --save_dir <dir> \
  --log_section fisher --log_section planner \
  --log_min_level DEBUG

# Suppress all but warnings
python demo.py --dataset replica --scene room_0 \
  --log_section all --log_min_level WARNING
```

## Dataset paths

Update `config/*.yaml` with local paths:

- Replica: download from the vMAP project (HuggingFace)
- ScanNet: official ScanNet repository
- Format: RGBD images + camera poses (4x4 matrices) + intrinsics

Results are written to `outputs/{scene}/` (rendered images and metrics).
