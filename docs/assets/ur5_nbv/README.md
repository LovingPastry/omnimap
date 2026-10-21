# UR5 主动视觉交互报告

打开 `docs/ur5_nbv_kinematics_report.html` 即可阅读；无需启动服务、安装依赖或联网。论文和代码外链需要网络。

首页“三环复盘”按 Tracking–Planning–Servo 解释具体改造顺序；完整源码依据和结果边界见仓库根目录 `ACTIVE_PERCEPTION_REVIEW.md`。论文精读保留为设计依据，真机方案尚未实现或验证。

## 更新

1. 修改正文原稿 `docs/ur5_nbv_kinematics_report.md`。
2. 修改本目录 `dashboard.html` 调整布局和交互。
3. 在安装了项目 ARIS 技能的工作区运行：

```bash
python3 docs/assets/ur5_nbv/build_report.py
```

构建脚本使用项目 `render-html` 技能的 Markdown 渲染器和自定义 dashboard 模板，将本目录图片以 data URI 嵌入输出。不要直接编辑生成的报告 HTML。HTML 保留正文原稿 SHA256、模板 SHA256 和生成时间；字体使用系统字体，不依赖 CDN。

## 交互与边界

- 论文分类与关键词搜索、原生详情展开、图片点击/键盘放大。
- 可调关节角的二维二连杆速度可操作椭圆演示；**不是 UR5 仿真或实验数据**。
- 完整六维位姿 / 五维位置与指向任务切换：展示局部满秩时的自由度计数，不代表全局安全可达。
- 明暗主题、滚动导航、阅读进度、打印/PDF。打印会临时展示全部论文及折叠详情，随后恢复阅读状态。
- 无 JavaScript 时保留正文、图片、原文链接及原生折叠阅读。

## 图片来源

图片均为论文原图，版权归原作者；正文图注保留出处。本地保存仅服务本研究报告的阅读与溯源。

| 本地文件 | 论文与图片 |
|---|---|
| `compass-overview.png` | Li 等，COMPASS，Fig. 1；https://arxiv.org/html/2509.14787v2/overview.png |
| `compass-hardware.png` | Li 等，COMPASS，Fig. 6；https://arxiv.org/html/2509.14787v2/real_world.png |
| `splat-cbf-kinova.png` | Khass 等，Splat-CBF，Fig. 9（机械臂仿真）；https://arxiv.org/html/2609.23100v1/cbfarms2.png |
| `oscbf-constraints.png` | Morton & Pavone，OSCBF，Fig. 1；https://arxiv.org/html/2503.06736v3/combined_fig.png |
| `reconstruction-focus.png` | Dursun 等，Object-Reconstruction-Aware Whole-body Control，Fig. 1；https://arxiv.org/html/2509.04094v3/Environment.png |

论文结论、代码诊断和方案建议的边界以正文为准。未进行 UR5 真机验证，未修改控制代码。论文核查截至 2026 年 10 月；源码行号基于报告生成时的仓库快照。

## 连续梯度方案（2026-10-21）

新增归一化最小奇异值、二维方向差分与闭式半空间修正。页面含合成谱/椭圆和向量投影两组交互，均非 UR5 实测。流程图源文件为 `gradient-planning.drawio`；页面用响应式等价流程展示。更新正文和模板后仍按上述脚本构建。
