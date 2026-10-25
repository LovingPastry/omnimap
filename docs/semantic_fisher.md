# 逐高斯目标语义 Fisher 筛选

`config/sim_rtabmap_config.yaml` 和 `config/rtabmap_config.yaml` 现已启用
`nbv_policy.fisher_mask.roi_query_mode: semantic`，查询文本默认 `cup`。
本次同步的是仓库中的真机配置和公共代码，没有启动真机运动。

## 数据与控制链

1. `OMNI.track()` 原有 YOLO-World → TAP → SBERT 流程按 `instance_skip` 更新语义体素和实例 embedding 表。
2. 3DGS 本帧优化、增密、剪枝完成后，`update_gaussian_semantics()` 对**当前全部高斯中心**做批量近邻查询，
   获得实例 ID。距离超过 `semantic_max_distance_m` 的高斯归为未知；默认使用 `instance.fill_dis`。
3. 查询文本只编码一次。对融合后的实例向量重新归一化后计算余弦相似度，再按高斯实例 ID 查表。
   无需为每个高斯存一份 384 维 embedding，也不额外运行检测器。零向量、未关联或不可靠实例不参与评分。
4. 生成 `gaussian_semantics`，包含查询文本、Tracking 帧号、逐高斯实例 ID 和相似度。
   每个 Tracking 帧重新关联，避免同样数量但位置/顺序已变化的高斯复用旧标签。
5. 快照 v4 将此 bundle 和高斯地图一起冻结、保存和加载。校验高斯数量及帧号，Planning 不加载 TSDF 或语义模型。
   v1–v3 快照仍可读；旧快照没有语义时，semantic 模式暂时停车等待新快照。
6. Planning 对 ID 有效、余弦相似度达到阈值的高斯赋权 1，其余赋权 0。
   多个满足阈值的实例都可参与，未通过三维包围盒做近似。
   每次梯度求解的所有探测视角使用同一组权重。

完整场景仍参与渲染与遮挡计算；信息增益按 `[xyz(N*3), opacity(N)]` 的现有贡献布局乘权后求和。
因此这一步控制的是目标函数的贡献范围，并不等于只渲染目标物体或消除全图渲染开销。

目标尚未积累足够语义观测或暂时失配时，不回退到全图/静态 ROI。
策略输出 `should_stop=true, stop_reason=semantic_target_unavailable`，使 Servo 的径向和朝向控制也停车。
这不锁存停止监视器，Tracking 继续积累观测，匹配恢复后可自动恢复规划。
仿真对此临时停车继续迭代，不把它当作实验终止。

## 配置与回退

```yaml
nbv_policy:
  fisher_mask:
    enabled: true
    roi_query: cup
    roi_query_mode: semantic
    roi_query_min_similarity: 0.35
    semantic_max_distance_m: null  # 使用 instance.fill_dis，当前为 0.05 m
    support_plane_z: null         # 若需要可额外手填桌面高度
```

`roi_bounds`、`roi_padding_m`、`roi_query_refresh_every` 不控制 semantic 模式；高斯与实例的关联每个 Tracking 帧更新。
`support_plane_z` 仍可叠加筛选，但没有增加自动桌面检测。

- 回退原版贡献范围：`fisher_mask.enabled: false`。代码的空配置仍默认关闭全部修复。
- 原来的语义包围盒对照：`roi_query_mode: sbert`。
- 临时文本重新检测出框的对照：`roi_query_mode: yolo_live`。
- 静态包围盒对照：`roi_query_mode: sbert, roi_query: null`，另设 `roi_bounds`。

后面三种模式仍保留原来的 ROI/回退语义，不能把它们与逐高斯语义筛选混为同一实验。
消融脚本使用自己的 `sim/ablation/sim_ablation_config.yaml`，保持原来的真值包围盒实验含义。

## 复现真实杯子场景的闭环验证

使用 InfoFlow 环境，在可访问 CUDA 和离屏 OpenGL 的环境中从仓库根目录执行：

```bash
python sim/prepare_rgbd_scene.py
python sim/sim_fisher_closed_loop.py \
  --pcd_path sim/sim_outputs/semantic_cup_scene/rgbd_scene.ply \
  --point_scale 1.0 --headless --cartesian --num_steps 30 \
  --save_dir sim/sim_outputs/semantic_cup_snapshot_final \
  --hemi_radius 0.5 --init_phi 0.6 --linear_vel_max 0.02 --dt 0.1 \
  --fisher_step_scale 0.03 --fisher_num_samples 8 --fisher_num_dense_points 16 \
  --nbv_set gradient_space=tangent \
  --nbv_set velocity_mapping.mode=normalized \
  --nbv_set velocity_mapping.v_nominal_mps=0.01 \
  --planner_snapshot_every 1 --save_frames --log_min_level INFO
python sim/check_semantic_fisher.py sim/sim_outputs/semantic_cup_snapshot_final
python tests/test_semantic_fisher.py
python tests/test_nbv_policy_improvements.py
```

场景来自仓库已有 `data/cup/imap/00` 的第 0 帧真实 RGB-D，保留杯子、桌面、纸张和周围背景。
只用深度有效范围构建彩色点云，没有给映射器提供目标 mask、实例 ID 或真值目标包围盒。
仿真相机运动后重新渲染 RGB-D，重新通过完整建图、语义融合和 Fisher 规划闭环。
该场景来自单视角，因此背面不可见区域有空洞，不能用于证明完整物体重建质量。

`--planner_snapshot_every 1` 每步真正保存、加载快照，并使用加载后的 FrozenPlannerBackend 规划。
设为 0 可测试普通 live backend；更大的正整数可测试 Planning 使用较低频率的地图快照。

逐步日志中的 `nbv_semantic_frame_idx` 应等于 `planning_snapshot_frame`；
它们可落后于相机/Planning tick，不应拿规划 tick 与 Tracking 帧号混为同一个时钟。
`nbv_semantic_status`、`nbv_semantic_known_ratio`、`nbv_semantic_target_ids`、`nbv_mask_kept_ratio`
分别用于检查语义就绪状态、已关联占比、选中实例和实际贡献占比。

## 验证证据与边界

2026-10-25，在 RTX 3090 上验证真实 RGB-D 衍生场景的 30 步闭环。
核验脚本要求实际命中目标、存在被排除的高斯、相机产生非零运动，并检查等待阶段的线速度与角速度均为零。
它还在最终真实 CUDA 高斯地图上检查：

- 全部非目标 Fisher 贡献严格为零，目标分数等于完整贡献向量的加权和。
- 把非目标原始 Fisher 项放大 1000 倍，目标分数、视角梯度和下一步位姿保持一致。
- Planning 后端不包含 live TSDF，语义与几何的帧号、长度一致。

数值结果保存在 [semantic_fisher_validation.json](semantic_fisher_validation.json)。
最终版本的 30 步中，前 3 步暂时停车，第 4 步起连续 27 步命中；实际选中比例为 11.63%–11.80%。
末帧 38,150 个高斯中选中 4,467 个、排除 33,683 个，语义帧和快照帧均为 29。
该帧完整评分为 11,426.71，目标评分为 1,385.29；二者只用于核验贡献筛选，不代表重建质量。
实际相机轨迹长约 1.39 cm。由于建图存在随机性，复跑数值可以不同。

单元测试另覆盖框内背景排除、批量距离门限、未知标签、融合向量归一化、增删/重排/移动高斯后的重新关联、
暂时停车与恢复、真实高斯快照的冻结/往返/旧版本兼容，以及帧号和长度不一致的拒绝。

这些验证证明数据链和目标函数筛选已接通，不证明检测/分割本身的准确率，也不证明重建优于基线或真机效果。
最近邻标签传播仍可能在边界误关联；阈值 0.35 和距离 0.05 m 需要结合实际场景评估。
旧彩色条纹资产的额外 12 步检查曾在匹配后发生语义漂移并正确等待，说明原有语义前端的稳定性仍是实际限制。
