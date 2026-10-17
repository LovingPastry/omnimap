# Scripts

环境激活与构建脚本。所有脚本都通过 `${BASH_SOURCE[0]}` 自行推断仓库根目录，
因此从任意工作目录调用都可以。

| 脚本 | 用途 |
| --- | --- |
| `source_env.sh` | 激活 `InfoFlow` conda 环境 + ROS Noetic + `ros_ws/devel`，算力侧使用 |
| `source_servo_env.sh` | 执行侧轻量环境，不激活重型 conda 环境，详见 [../docs/execution_side_setup.md](../docs/execution_side_setup.md) |
| `build_ros_ws.sh` | 用系统 Python 构建 `ros_ws`（会剥离 conda 环境变量，避免 catkin 混用解释器） |
| `reinstall_diff_grussian_rasterization.sh` | 重新编译安装 `modified-diff-gaussian-rasterization`，带调试符号 |

## 用法

```bash
# 环境脚本需要 source（会修改当前 shell 环境）
source scripts/source_env.sh
source scripts/source_servo_env.sh

# 构建脚本直接执行
./scripts/build_ros_ws.sh
./scripts/reinstall_diff_grussian_rasterization.sh
```

## Git 网络

本机直连 GitHub 的 22 端口被阻断，已在 `~/.ssh/config` 中让 `github.com` 经
Clash Verge 混合端口（`127.0.0.1:7897`）做 HTTP CONNECT 隧道，因此 `git push` /
`git fetch` 不需要额外参数。

Clash 未运行时会连不上，两种应急方式：

```bash
# 1) 临时改用 GitHub 的 443 端口 SSH 入口
git -c url."ssh://git@ssh.github.com:443/".insteadOf="git@github.com:" push origin master

# 2) 或在 ~/.ssh/config 里注释掉 ProxyCommand，改为
#      HostName ssh.github.com
#      Port 443
```

注意 `git config http.proxy` 只影响 HTTPS 传输，对 `git@github.com:` 这样的 SSH
远程无效——SSH 的代理必须通过 `ProxyCommand` 配置。

## 注意

`source_env.sh` 中的 `ROS_MASTER_URI` / `ROS_HOSTNAME` / `ROS_IP` 是写死的局域网地址，
换网络环境时需要改。`reinstall_diff_grussian_rasterization.sh` 里的 `ENV_PY` 与
`SITE_PACKAGES` 指向固定的 `FisherField` conda 环境路径，换环境时需要改。
