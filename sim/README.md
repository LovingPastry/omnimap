# sim 模块使用说明

`sim/` 是点云驱动的闭环仿真模块，用于验证 Fisher 主动视角规划。

## 推荐入口（简化 CLI）

日常运行请使用：

```bash
python3 sim/main.py ...
```

`sim/main.py` 只保留两种模式：

- 默认：无 GUI（仍会计算 Fisher、打印日志、保存产物）
- `--vis_gui`：开启 GUI，默认分离热力图窗口和速度场窗口

控制逻辑固定为：

- 笛卡尔控制（Cartesian）
- 角速度控制始终开启

## 常用参数（仅这 8 个）

`sim/main.py` 仅暴露以下常用参数：

1. `--pcd_path`
2. `--save_dir`
3. `--radial_gain`
4. `--angular_gain`
5. `--grad_eps`
6. `--dt`
7. `--linear_vel_max`
8. `--angular_speed_max`

## 最小命令

默认无 GUI：

```bash
python3 sim/main.py \
  --pcd_path replica/log_pcd_0/对比结果/RoboSeg_geo/Kettle.ply \
  --save_dir sim/sim_outputs/phase4_main \
  --radial_gain 0.2 \
  --angular_gain 2.0 \
  --grad_eps 0.01 \
  --dt 0.1 \
  --linear_vel_max 0.5 \
  --angular_speed_max 1.0
```

开启 GUI（split 双窗口）：

```bash
python3 sim/main.py \
  --pcd_path replica/log_pcd_0/对比结果/RoboSeg_geo/Kettle.ply \
  --save_dir sim/sim_outputs/phase4_main_gui \
  --radial_gain 0.2 \
  --angular_gain 2.0 \
  --grad_eps 0.01 \
  --dt 0.1 \
  --linear_vel_max 0.5 \
  --angular_speed_max 1.0 \
  --vis_gui
```

## 输出产物

每次运行都会输出并保存：

- `loop_log.jsonl`
- `loop_debug.csv`
- `trajectory_c2w_last.npy`
- `nbv_vis/*`（Fisher 相关导出）

## 高级入口（保留所有功能）

高级调参与实验入口保留在：

```bash
python3 sim/sim_fisher_closed_loop.py ...
```

该入口仍保留完整高级参数（如 headless / 采样密度 / 可视化细节开关），用于研究场景与深度调试。
