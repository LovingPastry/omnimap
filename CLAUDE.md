# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

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
source scripts/source_env.sh
```

### Build Commands

```bash
# Build ROS workspace (uses system Python, not conda)
./scripts/build_ros_ws.sh

# Reinstall diff-gaussian-rasterization after CUDA changes
./scripts/reinstall_diff_grussian_rasterization.sh
```

### Running the System

Entry-point invocations, the unified logging flags, and dataset path setup live in the
`omnimap-run` skill (`.claude/skills/omnimap-run/SKILL.md`) — invoke it when running
offline mapping, the simulation closed loop, or the online ROS node.

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
- See `docs/velocity_cmd_algorithm.md` for mathematical formulation

## Git Workflow

After any code changes in this workspace, run:
```bash
git add . && git commit -m "<commit>" && git push
```

This is a mandatory step per `AGENTS.md`.

## Additional Documentation

- `README.md`: Installation instructions, dataset preparation, citation
- `AGENTS.md`: Agent-specific rules, ROS focus, ARIS Codex integration
- `velocity_cmd_algorithm.md`: Mathematical formulation of velocity control from Fisher field
- `archives/README.md`: Explanation of archived files

