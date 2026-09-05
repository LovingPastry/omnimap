# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

OmniMap is an online 3D mapping framework that integrates three modalities:
- **Optical**: Photo-realistic appearance via 3D Gaussian Splatting (3DGS)
- **Geometric**: Precise layout via voxel-based TSDF fusion
- **Semantic**: Open-vocabulary understanding via YOLO-World and TAP models

The system supports real-time incremental mapping, active perception planning via Fisher information fields, and closed-loop robotic control.

## Development Environments

This project uses **multiple separate environments**:

1. **omnimap** conda environment: Core mapping (3DGS, TSDF, semantic models)
   - PyTorch 2.1.2 + CUDA 11.8
   - Custom CUDA extensions in `thirdparty/`
   - Activate: `conda activate omnimap`

2. **InfoFlow** conda environment: ROS integration (tracking, planning, servo)
   - Used for `info_flow/` nodes
   - Activate: `conda activate InfoFlow`

3. **System Python 3**: ROS workspace build
   - Uses system Python at `/usr/bin/python3`
   - ROS Noetic at `/opt/ros/noetic`

## Common Commands

### Environment Setup

```bash
# Activate main mapping environment
conda activate omnimap

# Set CUDA environment (required every session or add to conda activate script)
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$LD_LIBRARY_PATH

# Source ROS + InfoFlow environment for online nodes
source source_env.sh
```

### Build Commands

```bash
# Build ROS workspace (uses system Python, not conda)
./build_ros_ws.sh

# Reinstall diff-gaussian-rasterization after CUDA changes
./reinstall_diff_grussian_rasterization.sh
```

### Running the System

```bash
# Offline incremental mapping on datasets
python demo.py --dataset replica --scene room_0 [--vis_gui] [--start N] [--length M]
python demo.py --dataset scannet --scene scene0000_00

# Generate mesh from rendered outputs
python tsdf_integrate.py --dataset replica --scene room_0

# Simulation closed-loop (Fisher information-guided)
python sim/main.py \
  --pcd_path <path_to_ply> \
  --save_dir sim/sim_outputs/test \
  --fisher_step_scale 1e-4 \
  --radial_gain 0.2 \
  --angular_gain 2.0 \
  [--vis_gui]

# Online ROS node (three-loop architecture: tracking + planning + servo)
python info_flow/info_flow_node.py --config config/rtabmap_config.yaml
```

### Logging Control

Entry scripts (`demo.py`, `sim/main.py`, `info_flow/*.py`) support unified logging:

```bash
# Log sections: main, tsdf, gaussian, fisher, planner, profile, all
# Log levels: DEBUG, INFO, WARNING, ERROR
# Log profiles: quiet, default, debug

# Example: only main + planner sections at INFO level
python info_flow/info_flow_node.py \
  --config config/rtabmap_config.yaml \
  --log_section main --log_section planner \
  --log_min_level INFO

# Example: Fisher + Planner debug
python sim/main.py --pcd_path <path> --save_dir <dir> \
  --log_section fisher --log_section planner \
  --log_min_level DEBUG

# Example: suppress all but warnings
python demo.py --dataset replica --scene room_0 \
  --log_section all --log_min_level WARNING
```

## Code Architecture

### Core Mapping Pipeline (omnimap/)

**Entry point**: `omnimap/omni.py` → `OMNI` class
- Orchestrates TSDF + 3DGS backends frame-by-frame
- Main method: `OMNI.track(tstamp, image, depth, pose, ...)`

**Backends**:
- `tsdf_backend.py`: Voxel-based geometric mapping, spatial bounds filtering
- `gs_backend.py`: 3DGS optical mapping with adaptive camera modeling
  - Uses `omnimap/gaussian/` for renderer, scene representation, SLAM integration
  - Fisher information field computation in `omnimap/gaussian/renderer/nbv/`

**Visual module**: `visual_module.py`
- Semantic segmentation via YOLO-World + TAP
- Instance fusion and probabilistic matching

### ROS Integration (info_flow/)

**Three-loop architecture** (primary focus for ROS development):

1. **Tracking loop**: `info_flow_tracking_node.py`
   - SLAM frontend (RTABMap keyframe gating)
   - Publishes poses and keyframes

2. **Planning loop**: `info_flow_planning_node.py`
   - Active perception via Fisher information gradients
   - Generates velocity commands (linear + angular)

3. **Servo loop**: `info_flow_servo_runtime.py`
   - Low-level control execution
   - Interfaces with robot hardware (UR5)

**Unified node**: `info_flow_node.py`
- Single-process integration of tracking + planning
- Subscribes to compressed RGBD from ROS topics
- Publishes `TwistStamped` velocity commands

**Key components**:
- `slam_frontend.py`: RTABMap keyframe gate, queue management
- `planner_snapshot.py`: Snapshot state for planning decisions
- `distributed_common.py`: Shared utilities for node communication

### Simulation (sim/)

**Entry**: `sim/main.py` → delegates to `sim/sim_fisher_closed_loop.py`
- Offline closed-loop simulation using Fisher information field
- Cartesian + angular velocity control
- Visualization of Fisher heatmap and velocity field

### Configuration (config/)

- `replica_config.yaml`: Replica dataset parameters
- `scannet_config.yaml`: ScanNet dataset parameters
- `rtabmap_config.yaml`: Online ROS node configuration
- `sim_rtabmap_config.yaml`: Simulation configuration

**Key config sections**:
- `Training`: 3DGS optimization parameters
- `opt_params`: Learning rates and thresholds
- `tsdf`: Voxel size, block resolution, spatial bounds
- `instance`: Semantic instance fusion parameters
- `path`: Model weights and dataset paths

### Third-party Components (thirdparty/)

**Compiled extensions** (require `--no-build-isolation`):
- `simple-knn/`: K-nearest neighbors for point clouds
- `diff-gaussian-rasterization/`: Differentiable 3DGS renderer
- `lietorch/`: Lie algebra operations
- `mmcv/`: OpenMMLab computer vision library (v2.1.0 required)

**Submodules**:
- `YOLO-World/`: Open-vocabulary object detection
- `TAP/`: Tokenize Anything segmentation model
- `all-MiniLM-L6-v2/`: Sentence embeddings for semantic matching
- `mmyolo/`: Modified copy with relaxed mmcv version constraint

## Important Implementation Notes

### CUDA and Build System

- All custom CUDA extensions must be built with PyTorch 2.1.2 + CUDA 11.8
- The `$CONDA_PREFIX` must have CUDA toolkit installed
- Use `--no-build-isolation` when installing thirdparty packages
- ROS workspace must be built with system Python (`/usr/bin/python3`), not conda

### Model Weights

Required pretrained models in `weights/`:
- `yolo-world/yolo_world_v2_l_clip_large_o365v1_goldg_pretrain_800ft-9df82e55.pth`
- `tokenize-anything/tap_vit_l_v1_1.pkl`
- `tokenize-anything/merged_2560.pkl`
- `sbert/all-MiniLM-L6-v2/` (HuggingFace model directory)

Update paths in config files under `path:` section.

### Spatial Bounds Filtering

The TSDF backend supports spatial bounds to constrain mapping volume:
- Configure in `config/*.yaml` under `tsdf.spatial_bounds: [x_min, x_max, y_min, y_max, z_min, z_max]`
- Automatically enables filtering when bounds are provided
- Pixels outside bounds have depth/RGB zeroed before 3DGS input

### Version Compatibility Issues

**Critical version requirements**:
- `mmcv==2.1.0` (not 2.0.0 or 2.2.0)
- `transformers==4.36.2` (4.41+ breaks with PyTorch 2.1.2)
- Python 3.8-3.10 (PyTorch 2.1.2 compatibility)

**Known fixes**:
- YOLO-World syntax error at `yolo_world/models/detectors/yolo_world.py:61`: Change `self.text_feats, None = ...` to `self.text_feats, _ = ...`
- If seeing mmcv version assertion error, edit the library's `__init__.py` to accept 2.1.0

### Fisher Information Field

- Computed from 3DGS parameter uncertainties (mean, covariance, opacity)
- Used for active perception planning (next-best-view)
- Gradients computed via finite differences (configurable `--grad_eps`)
- Velocity commands synthesized from tangential (Fisher gradient) + radial (sphere constraint) components
- See `velocity_cmd_algorithm.md` for mathematical formulation

### ROS Topics and Messages

**Subscribed** (in `info_flow_node.py`):
- `/cam_1/color/image_raw/compressed`: RGB images
- `/cam_1/aligned_depth_to_color/image_raw/compressed`: Aligned depth
- `/cam_1/camera_info`: Camera intrinsics
- `/rtabmap/odom`: Odometry poses

**Published**:
- `/omnimap/cmd_vel`: `TwistStamped` velocity commands
- `/tf`: Camera pose transforms

**Custom messages**: `ros_ws/src/omnimap_msgs/`

## Git Workflow

After any code changes in this workspace, run:
```bash
git add . && git commit -m "<commit>" && git push
```

This is a mandatory step per `AGENTS.md`.

## Directory Organization

### Root-level Scripts
- `demo.py`: Offline incremental mapping entry point
- `tsdf_integrate.py`: Mesh generation from rendered outputs

### Core Directories
- `omnimap/`: Core mapping library (TSDF, 3DGS, semantic fusion)
- `info_flow/`: ROS integration nodes (tracking, planning, servo)
- `sim/`: Closed-loop simulation framework
- `config/`: Configuration files for different scenarios
- `ros_ws/`: ROS workspace for custom messages
- `thirdparty/`: Third-party dependencies and CUDA extensions

### Data and Outputs (git-ignored)
- `data/`: COCO/LVIS annotation files for YOLO-World
- `weights/`: Pretrained model weights (YOLO-World, TAP, SBERT)
- `outputs/`: Generated mapping results per scene
- `replica/`: Dataset files and experimental results
- `videos/`: Demo videos and recordings
- `practice_reports/`: Monthly progress reports

### Archives
- `archives/`: Deprecated code and backup files
  - `deprecated_nodes/`: Legacy ROS nodes replaced by `info_flow/`
  - `configs_backup/`: Runtime-generated configuration snapshots

### Tool Configuration (keep in repo)
- `.agents/`, `.aris/`, `.claude/`: AI assistant configuration
- `.github/`: GitHub workflows

## Additional Documentation

- `README.md`: Installation instructions, dataset preparation, citation
- `AGENTS.md`: Agent-specific rules, ROS focus, ARIS Codex integration
- `velocity_cmd_algorithm.md`: Mathematical formulation of velocity control from Fisher field
- `archives/README.md`: Explanation of archived files

## Dataset Paths

Update `config/*.yaml` with your local paths:
- Replica: Download from vMAP project (HuggingFace)
- ScanNet: Official ScanNet repository
- Format: RGBD images + camera poses (4x4 matrices) + intrinsics

Results saved to `outputs/{scene}/` containing rendered images and metrics.
