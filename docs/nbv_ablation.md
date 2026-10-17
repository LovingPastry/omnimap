# 主动视角策略：四项修复与消融

实现：[`omnimap/gaussian/renderer/nbv/policy_improvements.py`](../omnimap/gaussian/renderer/nbv/policy_improvements.py)
（由 `FisherMotionPolicy` 的 `nbv_config` 接入）。配置：`config/*.yaml` 的 `nbv_policy` 节。

## 背景

原版策略直接跟随 FisherRF 信息增益的局部梯度。这个准则原本是为"在自由空间候选集里挑一张图"设计的，
不需要可行性、速度尺度和停止条件；拿来驱动受约束的连续运动时，这三样都成了必需，对应三类观察到的问题：

| 现象 | 根因 | 修复（开关） |
|---|---|---|
| 北极附近打转 | (θ,φ) 参数化下 θ 自由度按 cos²φ 塌陷 | `gradient_space: tangent`（`6b99a62` 已有） |
| 冲向半球边缘 / 桌下 | 打分对**全部**高斯求和，桌面/背景高斯和不可达视角的"信息"无法兑现却在拉相机；梯度不知道哪里可达 | `fisher_mask`（ROI + 支撑面可见性）+ `feasibility`（仰角带 / 工作空间切锥投影） |
| 速度尺度靠试、常撞限幅 | 梯度幅值直接当速度，同一 `fisher_step_scale` 在不同场景下速度差两个数量级 | `velocity_mapping: normalized` |
| 没有停止信号 | 只有 `spherical_speed_min`（真机为 0） | `stop`（饱和 / 覆盖 / 预算 / 停滞，取或） |

## 四项修复的定义

**1. 可行性 `feasibility`**：对每个约束（仰角下限、上限、可选的相机位置盒的六个面）给出外法向 `n` 和激活度
`s∈[0,1]`（离边界 `margin` 以内线性升到 1）。对指令切向速度做 `v ← v − s·max(0, v·n)·n`，只削掉"往外"的分量，
往里走不受影响；`hard_clamp` 再把积分出的位姿夹回仰角带。作用在所有输出分支共用的二维速率上，所以
cartesian / spherical_rate / angular 三条路径都继承。

**2. 可见性掩码 `fisher_mask`**：贡献向量 `cur/(history+λ)` 按 `[xyz(N·3), opacity(N)]` 布局逐元素乘权重：
ROI 盒外、支撑面以下的高斯权重为 0；相机低于支撑面时整幅打分乘以 sigmoid 门（被桌面挡住）。
这是用支撑面近似 TSDF 遮挡，不是逐高斯的 TSDF 光线投射。

**3. 速率映射 `velocity_mapping: normalized`**：梯度只给方向，速率与梯度绝对尺度无关：

```
direction = g / |g|
speed     = v_nominal · |g| / (|g| + κ),   κ = |g| 的滑动中位数（窗口 kappa_window）
```

把场整体放大 10⁴ 倍，指令完全不变（单元测试 `test_rate_mapper_is_scale_invariant`）；`v_nominal` 是有物理意义的
m/s，`linear_vel_max` 退回安全兜底。

**4. 停止判据 `stop`**（满足任一即停，触发后锁存）：
- A 信息饱和：Fisher 分数滑动均值 < `score_ratio` × 历史最大滑动均值
- B 视角球覆盖：仰角带内 Fibonacci 方向中，落在某个已访问视角 `coverage_radius_deg` 以内的比例 ≥ `coverage_threshold`
- C 预算：步数 / 路径长 / 时间（真机用墙钟）
- D 停滞：指令切向速度连续低于阈值（默认关）

覆盖判据用相机位姿而不是 TSDF 权重：当前 TSDF 后端不暴露逐体素积分权重。

## 配置与回退

`nbv_policy` 全部关闭（或整节缺省）= 原版实现。回归验证：默认配置与改动前 HEAD 的 `motion_policy`
在 angle/tangent × cartesian/spherical_rate/angular 共 447 步上**逐位一致**。

```yaml
nbv_policy:
  gradient_space: tangent          # angle = 原版
  feasibility:      {enabled: true, phi_min_deg: 15, phi_max_deg: 80}
  fisher_mask:      {enabled: true, roi_bounds: tsdf, support_plane_z: tsdf_min_z}
  velocity_mapping: {mode: normalized, v_nominal_mps: 0.03}
  stop:             {enabled: true}
```

- `roi_bounds: tsdf` / `support_plane_z: tsdf_min_z` 取 `tsdf.spatial_bounds`；仿真里可用 `object` / `object_min_z`（物体 AABB）。
- 仿真命令行覆盖：`--nbv_config file.yaml`、`--nbv_set fisher_mask.enabled=true`（可重复）；`--gradient_space` 仍有效且优先。
- 真机两个规划入口（`info_flow_planning_node.py`、`info_flow_node.py`）读取同一节。**真机侧只同步了代码，未上机验证**；
  真机配置里窗口按 30 Hz 规划 tick 计（`score_window: 300`、`min_steps: 600`），预算用墙钟。

## 运行消融

```bash
bash sim/ablation/run_nbv_ablation.sh sim/sim_outputs/nbv_ablation 300 2   # 输出目录 步数 并行数（每个约 7.6 GB 显存）
python sim/ablation/analyze_nbv_ablation.py sim/sim_outputs/nbv_ablation
python tests/test_nbv_policy_improvements.py                              # CPU/GPU 均可，无需 pytest
```

需要 InfoFlow 环境（含 `modified_diff_gaussian_rasterization`）；omnimap 环境里没有这个包。
