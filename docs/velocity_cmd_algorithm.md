# UR5 VLM Field Runner 速度指令算法说明

本文档描述如何从策略场生成 `Twist` 命令。

> 注：文中原先引用的 `ros1_nodes/ur5_vlm_field_runner.py` 已不在仓库中。当前对应实现分布在
> [omnimap/gaussian/renderer/nbv/motion_policy.py](../omnimap/gaussian/renderer/nbv/motion_policy.py)（策略场与角速度积分）、
> [info_flow/info_flow_planning_node.py](../info_flow/info_flow_planning_node.py)（在线规划环）与
> [sim/sim_fisher_closed_loop.py](../sim/sim_fisher_closed_loop.py)（仿真闭环）。
> 下面的公式仍然描述该控制律本身。

- 线速度：`[v_x, v_y, v_z]`
- 角速度：`[\omega_x, \omega_y, \omega_z]`

最终输出：

$$
\mathbf{u}
=
\begin{bmatrix}
v_x & v_y & v_z & \omega_x & \omega_y & \omega_z
\end{bmatrix}^{\top}
\in \mathbb{R}^{6}
$$

## 1. 先决条件与停机条件

若任一条件不满足，直接输出零速度：

1. 目标对象尚未确定：`target_object_curr is None`
2. NBV 场为空：`len(nbv_fields) == 0`

$$
\mathbf{u}=\mathbf{0}_{6}
$$

## 2. 切向速度分量（来自 Policy）

在当前相机位置 `x.translation` 处查询融合切向场：

$$
\mathbf{e}_t
=
\mathrm{FieldFusion}(\mathbf{p})
,\quad
\mathbf{p}=x.\mathrm{translation}
$$

该向量由 `policy.query_field_fusion_from_list(...)` 返回，已是笛卡尔坐标系下的 3D 向量。

## 3. 法向（径向）修正分量

定义球心与半径：

$$
\mathbf{c}=\mathrm{view\_sphere.center},
\quad
R=\mathrm{view\_sphere.r}
$$

定义当前半径距离：

$$
r=\|\mathbf{p}-\mathbf{p}_0\|_2
$$

其中 $\mathbf{p}_0$ 对应代码中的 `pcd_shift`。

径向修正项：

$$
\mathbf{e}_n
=
(\mathbf{p}-\mathbf{c})\cdot\frac{R-r}{r+\varepsilon},
\quad
\varepsilon=10^{-6}
$$

当相机位于球内时（`r<R`），该项把相机推向目标球面；球外时该项不参与线速度合成。

## 4. 线速度合成与限幅

线速度的未限幅合成：

$$
\mathbf{v}_{raw}
=
1.0\cdot\mathbf{e}_t
+
2.0\cdot\mathbb{I}(r<R)\cdot\mathbf{e}_n
$$

其中 $\mathbb{I}(\cdot)$ 为指示函数。

设线速度上限为 $v_{max}=\mathrm{linear\_vel}$ ，则限幅后：

$$
\mathbf{v}
=
\begin{cases}
\mathbf{0}, & \|\mathbf{v}_{raw}\|_2 \le 10^{-9} \\
\mathbf{v}_{raw}\cdot\frac{\min(\|\mathbf{v}_{raw}\|_2,v_{max})}{\|\mathbf{v}_{raw}\|_2}, & \text{otherwise}
\end{cases}
$$

## 5. 角速度计算（注视球心）

先由当前位置对应球坐标角 `(theta, phi)` 生成期望相机姿态：

$$
\mathbf{R}_{des}
=
\mathrm{ViewSphere.get\_view}(\theta,\phi).\mathrm{rotation}
$$

当前姿态：

$$
\mathbf{R}_{cur}=x.\mathrm{rotation}
$$

姿态误差旋转：

$$
\mathbf{R}_{err}
=
\mathbf{R}_{des}\mathbf{R}_{cur}^{-1}
$$

设角速度增益为 $k_{\omega}=\mathrm{angular\_vel}$ ，则角速度命令：

$$
\boldsymbol{\omega}
=
k_{\omega}\cdot\mathrm{rotvec}(\mathbf{R}_{err})
$$

若关闭角速度输出（`enable_angular=False`）：

$$
\boldsymbol{\omega}=\mathbf{0}_{3}
$$

## 6. 最终 Twist 输出

$$
\mathbf{u}
=
\begin{bmatrix}
\mathbf{v} \\
\boldsymbol{\omega}
\end{bmatrix}
\in \mathbb{R}^{6}
$$

对应代码返回：

```python
return np.r_[linear, angular]
```
