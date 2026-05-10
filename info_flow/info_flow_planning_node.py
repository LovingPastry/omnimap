from __future__ import annotations

import argparse
import threading
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import rospy
import torch
from geometry_msgs.msg import PoseStamped

from distributed_common import (
    CachedPose,
    CachedSnapshotRef,
    apply_fixed_hemisphere_reference,
    c2w_from_pose_stamped,
    c2w_to_w2c_posevec,
    configure_entry_logging,
    fixed_hemisphere_from_config,
    import_omnimap_msgs,
    load_runtime_config,
    set_nofile_limit,
    wait_for_camera_calibration,
)
from omnimap.gaussian.renderer.nbv.motion_policy import FisherMotionPolicy
from omnimap.util.utils import get_section_logger
from planner_snapshot import PlannerSnapshot, load_planner_snapshot_file


@dataclass
class PlanningStats:
    pose_updates: int = 0
    snapshot_ref_updates: int = 0
    snapshot_load_success: int = 0
    snapshot_load_fail: int = 0
    planner_steps: int = 0
    planner_nonzero: int = 0
    planner_stop_missing_pose: int = 0
    planner_stop_missing_snapshot: int = 0
    planner_stop_pose_stale: int = 0
    planner_stop_snapshot_load: int = 0
    planner_stop_policy: int = 0
    planner_exception: int = 0


class InfoFlowPlanningNode:
    def __init__(self, args, config):
        rospy.init_node("info_flow_planning_node", anonymous=True)
        _, SphericalCommandMsg = import_omnimap_msgs()

        self.args = args
        self.config = config
        self.SphericalCommandMsg = SphericalCommandMsg
        self.main_logger = get_section_logger("entry.infoflow_planning", "main")
        self.planner_logger = get_section_logger("planner.infoflow_planning", "planner")
        self.profile_logger = get_section_logger("profile.infoflow_planning", "profile")
        self.world_frame = str(getattr(args, "world_frame", ""))
        self.planner_hz = float(args.planner_hz)
        self.pose_stale_timeout_sec = float(args.pose_stale_timeout_sec)
        # 使用双阈值 stale 判定；按需求将阈值设置得非常大，尽量避免因短时时延触发停车。
        self.pose_stamp_stale_timeout_sec = 1e6
        self.pose_receipt_stale_timeout_sec = 1e6
        self.status_log_interval_sec = max(0.2, float(args.status_log_interval_sec))

        self.main_logger.info("正在等待相机内参消息...")
        _, K, calib, image_size_hw = wait_for_camera_calibration(args.camera_info_topic)
        self.K = K
        self.calib = calib
        self.image_size_hw = image_size_hw

        self.motion_policy = FisherMotionPolicy(
            fisher_step_scale=args.fisher_step_scale,
            cartesian=True,
            dt=args.dt,
            radial_gain=args.radial_gain,
            linear_vel_max=args.linear_vel_max,
            angular_gain=args.angular_gain,
            angular_speed_max=args.angular_speed_max,
            enable_angular=args.enable_angular,
            grad_eps=args.grad_eps,
            spherical_speed_min=args.spherical_speed_min,
            planner_output_mode="spherical_rate",
            verbose=True,
        )
        self.motion_policy_lock = threading.Lock()
        self.fixed_hemisphere_center, self.fixed_hemisphere_radius_m = (
            fixed_hemisphere_from_config(config)
        )
        (
            self.reference_center_mode,
            self.reference_center_online_enabled,
            self.reference_center_ema_alpha,
            self.reference_bounds_min,
            self.reference_bounds_max,
            self.reference_radius_source,
        ) = self._reference_runtime_settings_from_config(config)
        if self.fixed_hemisphere_center is not None:
            self.main_logger.warning(
                "参考球策略：center_init=spatial_bounds_center=%s radius=%.3fm source=%s center_mode=%s center_online=%s ema_alpha=%.3f",
                self.fixed_hemisphere_center.tolist(),
                float(self.fixed_hemisphere_radius_m),
                self.reference_radius_source,
                self.reference_center_mode,
                bool(self.reference_center_online_enabled),
                float(self.reference_center_ema_alpha),
            )
            apply_fixed_hemisphere_reference(
                self.motion_policy,
                self.fixed_hemisphere_center,
                self.fixed_hemisphere_radius_m,
            )

        self.pose_lock = threading.Lock()
        self.latest_pose: Optional[CachedPose] = None
        self.snapshot_ref_lock = threading.Lock()
        self.latest_snapshot_ref: Optional[CachedSnapshotRef] = None
        self.loaded_snapshot: Optional[PlannerSnapshot] = None
        self.loaded_snapshot_version: int = -1
        self.snapshot_load_failed_version: int = -1
        self.snapshot_load_failed_reason: str = ""
        self.spherical_cmd_pub = rospy.Publisher(
            args.spherical_cmd_topic,
            SphericalCommandMsg,
            queue_size=1,
        )

        self.pose_sub = rospy.Subscriber(
            args.pose_topic, PoseStamped, self.pose_callback, queue_size=1
        )
        SnapshotRefMsg, _ = import_omnimap_msgs()
        self.snapshot_ref_sub = rospy.Subscriber(
            args.snapshot_ref_topic,
            SnapshotRefMsg,
            self.snapshot_ref_callback,
            queue_size=1,
        )
        self.timer = rospy.Timer(
            rospy.Duration(1.0 / max(self.planner_hz, 1e-6)),
            self._planning_timer_callback,
        )

        self.stats_lock = threading.Lock()
        self.stats = PlanningStats()
        self.planner_tick = 0
        self._last_status_wall = float(time.monotonic())
        self._last_status_steps = 0
        self._last_status_nonzero = 0
        self._throttle_last = {}

        self.main_logger.info(
            "Planning 节点已启动：pose_topic=%s snapshot_ref_topic=%s spherical_cmd_topic=%s planner_hz=%.1f stale(T1=%.1fs,T2=%.1fs)",
            args.pose_topic,
            args.snapshot_ref_topic,
            args.spherical_cmd_topic,
            float(self.planner_hz),
            float(self.pose_stamp_stale_timeout_sec),
            float(self.pose_receipt_stale_timeout_sec),
        )

    def _planner_log_throttle(
        self, key: str, interval_sec: float, level: str, msg: str, *args
    ):
        now = float(time.monotonic())
        last = float(self._throttle_last.get(key, -1e18))
        if (now - last) < float(interval_sec):
            return
        self._throttle_last[key] = now
        getattr(self.planner_logger, str(level).lower())(msg, *args)

    @staticmethod
    def _reference_runtime_settings_from_config(
        config: dict,
    ) -> Tuple[str, bool, float, Optional[np.ndarray], Optional[np.ndarray], str]:
        tsdf_cfg = config.get("tsdf", {}) if isinstance(config, dict) else {}
        bounds = tsdf_cfg.get("spatial_bounds", None) if isinstance(tsdf_cfg, dict) else None
        use_adaptive_radius = bool(tsdf_cfg.get("reference_radius_use_adaptive", False))
        center_mode = str(tsdf_cfg.get("reference_center_mode", "dynamic")).strip().lower()
        if center_mode not in {"fixed", "dynamic"}:
            center_mode = "dynamic"
        center_online_cfg = bool(tsdf_cfg.get("reference_center_online_update", True))
        center_online = bool(center_online_cfg and center_mode == "dynamic")
        ema_alpha = float(tsdf_cfg.get("reference_center_ema_alpha", 0.1))
        ema_alpha = float(np.clip(ema_alpha, 0.0, 1.0))
        if isinstance(bounds, (list, tuple)) and len(bounds) == 6:
            x_min, x_max, y_min, y_max, z_min, z_max = [float(v) for v in bounds]
            bmin = np.array([x_min, y_min, z_min], dtype=np.float64)
            bmax = np.array([x_max, y_max, z_max], dtype=np.float64)
            return (
                center_mode,
                center_online,
                ema_alpha,
                bmin,
                bmax,
                ("adaptive" if use_adaptive_radius else "default"),
            )
        return center_mode, center_online, ema_alpha, None, None, "default_no_bounds"

    @staticmethod
    def _extract_live_center_from_backend(snapshot_backend) -> Optional[np.ndarray]:
        tsdfs = getattr(snapshot_backend, "tsdfs", None)
        if tsdfs is None or not hasattr(tsdfs, "get_pointcloud_center"):
            return None
        try:
            center = tsdfs.get_pointcloud_center()
        except Exception:
            return None
        if center is None:
            return None
        if isinstance(center, torch.Tensor):
            center_np = center.detach().cpu().numpy().astype(np.float64).reshape(3)
        else:
            center_np = np.asarray(center, dtype=np.float64).reshape(3)
        if not np.isfinite(center_np).all():
            return None
        return center_np

    def _update_reference_center_online(self, live_center: np.ndarray) -> None:
        if self.fixed_hemisphere_center is None:
            return
        alpha = float(self.reference_center_ema_alpha)
        if alpha <= 0.0:
            return
        prev_center = np.asarray(self.fixed_hemisphere_center, dtype=np.float64).reshape(3)
        new_center = (1.0 - alpha) * prev_center + alpha * np.asarray(
            live_center, dtype=np.float64
        ).reshape(3)
        if self.reference_bounds_min is not None and self.reference_bounds_max is not None:
            new_center = np.minimum(
                np.maximum(new_center, self.reference_bounds_min),
                self.reference_bounds_max,
            )
        self.fixed_hemisphere_center = new_center
        self._planner_log_throttle(
            "reference_center_update",
            1.0,
            "info",
            "reference_center_online: prev=[%.3f %.3f %.3f] live=[%.3f %.3f %.3f] new=[%.3f %.3f %.3f]",
            float(prev_center[0]),
            float(prev_center[1]),
            float(prev_center[2]),
            float(live_center[0]),
            float(live_center[1]),
            float(live_center[2]),
            float(new_center[0]),
            float(new_center[1]),
            float(new_center[2]),
        )

    def pose_callback(self, msg: PoseStamped) -> None:
        c2w = c2w_from_pose_stamped(msg)
        pose_w2c = c2w_to_w2c_posevec(c2w)
        pose = CachedPose(
            stamp=msg.header.stamp,
            pose_w2c=pose_w2c,
            pose_4x4=c2w,
            wall_time=float(time.monotonic()),
        )
        with self.pose_lock:
            self.latest_pose = pose
        with self.stats_lock:
            self.stats.pose_updates += 1
        self.profile_logger.debug(
            "pose_rx: stamp=%.3f pose_age=%.3fs",
            float(msg.header.stamp.to_sec()),
            float((rospy.Time.now() - msg.header.stamp).to_sec()),
        )

    def snapshot_ref_callback(self, msg) -> None:
        snapshot_ref = CachedSnapshotRef(
            run_id=str(msg.run_id),
            model_version=int(msg.model_version),
            keyframe_idx=int(msg.keyframe_idx),
            snapshot_uri=str(msg.snapshot_uri),
            created_wall_time=float(msg.created_wall_time),
            runtime_device_hint=str(msg.runtime_device_hint),
            stamp=msg.header.stamp,
            receipt_wall_time=float(time.monotonic()),
        )
        with self.snapshot_ref_lock:
            self.latest_snapshot_ref = snapshot_ref
        with self.stats_lock:
            self.stats.snapshot_ref_updates += 1
        self.profile_logger.debug(
            "snapshot_ref_rx: model_v=%d frame=%d uri=%s",
            int(snapshot_ref.model_version),
            int(snapshot_ref.keyframe_idx),
            snapshot_ref.snapshot_uri,
        )

    def _ensure_latest_snapshot_loaded(self) -> Optional[PlannerSnapshot]:
        with self.snapshot_ref_lock:
            snapshot_ref = self.latest_snapshot_ref
        if snapshot_ref is None:
            return self.loaded_snapshot
        if (
            self.loaded_snapshot is not None
            and self.loaded_snapshot_version == snapshot_ref.model_version
        ):
            return self.loaded_snapshot

        load_t0 = time.perf_counter()
        try:
            snapshot = load_planner_snapshot_file(snapshot_ref.snapshot_uri)
            if int(snapshot.model_version) != int(snapshot_ref.model_version):
                raise ValueError(
                    f"snapshot version mismatch: file={snapshot.model_version} ref={snapshot_ref.model_version}"
                )
        except Exception as exc:
            self.snapshot_load_failed_version = int(snapshot_ref.model_version)
            self.snapshot_load_failed_reason = str(exc)
            with self.stats_lock:
                self.stats.snapshot_load_fail += 1
            self._planner_log_throttle(
                "snapshot_load_failure",
                1.0,
                "warning",
                "加载规划快照失败：model_v=%d uri=%s err=%s",
                int(snapshot_ref.model_version),
                snapshot_ref.snapshot_uri,
                exc,
            )
            return self.loaded_snapshot

        self.loaded_snapshot = snapshot
        self.loaded_snapshot_version = int(snapshot.model_version)
        self.snapshot_load_failed_version = -1
        self.snapshot_load_failed_reason = ""
        with self.stats_lock:
            self.stats.snapshot_load_success += 1
        self.profile_logger.info(
            "snapshot_load: model_v=%d frame=%d load_ms=%.2f uri=%s",
            int(snapshot.model_version),
            int(snapshot.keyframe_idx),
            float((time.perf_counter() - load_t0) * 1000.0),
            snapshot_ref.snapshot_uri,
        )
        return snapshot

    def _publish_spherical_command(
        self,
        *,
        stamp: rospy.Time,
        model_version: int,
        planner_tick: int,
        dt: float,
        theta_rate: float,
        phi_rate: float,
        reference_radius: float,
        reference_scene_center: np.ndarray,
        fisher_score: float,
        should_stop: bool,
        stop_reason: str,
    ) -> None:
        msg = self.SphericalCommandMsg()
        msg.header.stamp = stamp if stamp is not None else rospy.Time.now()
        msg.header.frame_id = self.world_frame
        msg.model_version = int(model_version)
        msg.planner_tick = int(planner_tick)
        msg.dt = float(dt)
        msg.theta_rate = float(theta_rate)
        msg.phi_rate = float(phi_rate)
        msg.reference_radius = float(reference_radius)
        msg.reference_scene_center = [
            float(reference_scene_center[0]),
            float(reference_scene_center[1]),
            float(reference_scene_center[2]),
        ]
        msg.fisher_score = float(fisher_score)
        msg.should_stop = bool(should_stop)
        msg.stop_reason = str(stop_reason)
        self.spherical_cmd_pub.publish(msg)

    def _publish_stop_command(
        self,
        *,
        reason: str,
        stamp: Optional[rospy.Time],
        model_version: int,
    ) -> None:
        self._publish_spherical_command(
            stamp=stamp if stamp is not None else rospy.Time.now(),
            model_version=model_version,
            planner_tick=self.planner_tick,
            dt=float(self.motion_policy.dt),
            theta_rate=0.0,
            phi_rate=0.0,
            reference_radius=float(
                getattr(self.motion_policy, "reference_radius", 0.0) or 0.0
            ),
            reference_scene_center=np.asarray(
                getattr(self.motion_policy, "reference_scene_center", np.zeros(3)),
                dtype=np.float64,
            ).reshape(3),
            fisher_score=0.0,
            should_stop=True,
            stop_reason=reason,
        )

    def _maybe_log_status(self) -> None:
        now = float(time.monotonic())
        elapsed = now - self._last_status_wall
        if elapsed < self.status_log_interval_sec:
            return
        with self.stats_lock:
            stats = PlanningStats(**self.stats.__dict__)
        d_steps = int(stats.planner_steps) - int(self._last_status_steps)
        d_nonzero = int(stats.planner_nonzero) - int(self._last_status_nonzero)
        self._last_status_wall = now
        self._last_status_steps = int(stats.planner_steps)
        self._last_status_nonzero = int(stats.planner_nonzero)
        self.planner_logger.info(
            (
                "Planning 状态：plan_hz=%.1f cmd_hz=%.1f 累计(steps=%d nonzero=%d "
                "missing_pose=%d missing_snapshot=%d pose_stale=%d snapshot_load_fail=%d "
                "policy_stop=%d exception=%d snapshot_ok=%d snapshot_fail=%d)"
            ),
            float(d_steps / max(elapsed, 1e-6)),
            float(d_nonzero / max(elapsed, 1e-6)),
            int(stats.planner_steps),
            int(stats.planner_nonzero),
            int(stats.planner_stop_missing_pose),
            int(stats.planner_stop_missing_snapshot),
            int(stats.planner_stop_pose_stale),
            int(stats.planner_stop_snapshot_load),
            int(stats.planner_stop_policy),
            int(stats.planner_exception),
            int(stats.snapshot_load_success),
            int(stats.snapshot_load_fail),
        )

    def _planning_timer_callback(self, _event):
        # 周期性规划入口：
        # 按固定 tick 执行一次“状态检查 -> 策略求解 -> 指令发布”。
        self.planner_tick += 1
        planning_t0 = time.perf_counter()
        with self.stats_lock:
            self.stats.planner_steps += 1

        # 读取最新位姿与最新 snapshot 引用（均为跨回调共享状态，需加锁）。
        with self.pose_lock:
            pose_state = self.latest_pose
        snapshot = self._ensure_latest_snapshot_loaded()
        with self.snapshot_ref_lock:
            snapshot_ref = self.latest_snapshot_ref

        # 无位姿：等待上游恢复；不主动发布 stop，避免初始化阶段高频 0 速刷屏。
        if pose_state is None:
            self._planner_log_throttle(
                "missing_pose",
                1.0,
                "warning",
                "规划等待 pose topic：%s",
                self.args.pose_topic,
            )
            with self.stats_lock:
                self.stats.planner_stop_missing_pose += 1
            self.profile_logger.info(
                "planning耗时：idx=%d total=%.2fms status=missing_pose",
                int(self.planner_tick),
                float((time.perf_counter() - planning_t0) * 1000.0),
            )
            self._maybe_log_status()
            return

        # snapshot 加载曾失败且版本未变化：等待版本更新重试；不主动发布 stop。
        if snapshot_ref is not None and self.snapshot_load_failed_version == int(
            snapshot_ref.model_version
        ):
            with self.stats_lock:
                self.stats.planner_stop_snapshot_load += 1
            self.profile_logger.info(
                "planning耗时：idx=%d total=%.2fms status=snapshot_load_failure",
                int(self.planner_tick),
                float((time.perf_counter() - planning_t0) * 1000.0),
            )
            self._maybe_log_status()
            return

        # 尚无可用地图快照：等待 snapshot；不主动发布 stop，避免初始化阶段高频 0 速。
        if snapshot is None:
            self._planner_log_throttle(
                "missing_snapshot",
                1.0,
                "warning",
                "规划等待 snapshot ref：%s",
                self.args.snapshot_ref_topic,
            )
            with self.stats_lock:
                self.stats.planner_stop_missing_snapshot += 1
            self.profile_logger.info(
                "planning耗时：idx=%d total=%.2fms status=missing_snapshot",
                int(self.planner_tick),
                float((time.perf_counter() - planning_t0) * 1000.0),
            )
            self._maybe_log_status()
            return

        # 双重新鲜度检查：
        # - stamp_age: ROS 时间戳到当前时刻的延迟
        # - receipt_age: 本进程收到该位姿后的墙钟时间
        stamp_age = float((rospy.Time.now() - pose_state.stamp).to_sec())
        receipt_age = float(time.monotonic() - pose_state.wall_time)
        self.profile_logger.debug(
            "pose_age_check(planner): stamp_age=%.3fs receipt_age=%.3fs stamp=%.3f",
            float(stamp_age),
            float(receipt_age),
            float(pose_state.stamp.to_sec()),
        )
        # 位姿过期：宁可停车，不输出基于陈旧状态的运动指令。
        # 双阈值判定：stamp_age > T1 或 receipt_age > T2。
        if (stamp_age > self.pose_stamp_stale_timeout_sec) or (
            receipt_age > self.pose_receipt_stale_timeout_sec
        ):
            self._planner_log_throttle(
                "pose_stale",
                1.0,
                "warning",
                (
                    "Planning 位姿过期（stamp_age=%.3fs/T1=%.3fs receipt_age=%.3fs/T2=%.3fs），"
                    "发布 stop command。"
                ),
                stamp_age,
                self.pose_stamp_stale_timeout_sec,
                receipt_age,
                self.pose_receipt_stale_timeout_sec,
            )
            self.profile_logger.info(
                "pose_stale(planner): stamp_age=%.3fs receipt_age=%.3fs stamp=%.3f",
                float(stamp_age),
                float(receipt_age),
                float(pose_state.stamp.to_sec()),
            )
            self._publish_stop_command(
                reason="pose_stale",
                stamp=pose_state.stamp,
                model_version=int(snapshot.model_version),
            )
            with self.stats_lock:
                self.stats.planner_stop_pose_stale += 1
            self.profile_logger.info(
                "planning耗时：idx=%d total=%.2fms status=pose_stale",
                int(self.planner_tick),
                float((time.perf_counter() - planning_t0) * 1000.0),
            )
            self._maybe_log_status()
            return

        try:
            with self.motion_policy_lock:
                # 半球参考约束每个 tick 施加；球心可按在线估计做 EMA 更新。
                if self.reference_center_online_enabled:
                    live_center = self._extract_live_center_from_backend(snapshot.backend)
                    if live_center is not None:
                        self._update_reference_center_online(live_center)
                apply_fixed_hemisphere_reference(
                    self.motion_policy,
                    self.fixed_hemisphere_center,
                    self.fixed_hemisphere_radius_m,
                )
                # 用当前位姿 + 最新 GS 快照执行一步策略求解。
                motion_result = self.motion_policy.next_pose_from_c2w(
                    gs_backend=snapshot.backend,
                    current_c2w=np.asarray(pose_state.pose_4x4, dtype=np.float64),
                    intrinsics_vec=self.calib,
                    image_size=self.image_size_hw,
                    idx=self.planner_tick,
                )
        except Exception as exc:
            # 策略异常时立即降级为 stop command，避免异常传播为危险动作。
            self._planner_log_throttle(
                "policy_exception",
                1.0,
                "warning",
                "Fisher 策略规划异常，发布 stop command：%s",
                exc,
            )
            self._publish_stop_command(
                reason=f"policy_exception:{exc}",
                stamp=pose_state.stamp,
                model_version=int(snapshot.model_version),
            )
            with self.stats_lock:
                self.stats.planner_exception += 1
            self.profile_logger.info(
                "planning耗时：idx=%d total=%.2fms status=policy_exception",
                int(self.planner_tick),
                float((time.perf_counter() - planning_t0) * 1000.0),
            )
            self._maybe_log_status()
            return

        # 将策略输出统一封装为球坐标控制指令发布给 servo。
        reference_radius = float(getattr(motion_result, "reference_radius", 0.0))
        reference_scene_center = np.asarray(
            getattr(motion_result, "reference_scene_center", [0.0, 0.0, 0.0]),
            dtype=np.float64,
        ).reshape(3)
        self._publish_spherical_command(
            stamp=pose_state.stamp,
            model_version=int(snapshot.model_version),
            planner_tick=int(self.planner_tick),
            dt=float(getattr(motion_result, "dt", self.motion_policy.dt)),
            theta_rate=float(getattr(motion_result, "theta_rate_applied", 0.0)),
            phi_rate=float(getattr(motion_result, "phi_rate_applied", 0.0)),
            reference_radius=reference_radius,
            reference_scene_center=reference_scene_center,
            fisher_score=float(getattr(motion_result, "fisher_score", 0.0)),
            should_stop=bool(getattr(motion_result, "should_stop", False)),
            stop_reason=str(getattr(motion_result, "stop_reason", "unknown")),
        )
        self._planner_log_throttle(
            "reference_sphere_publish",
            1.0,
            "info",
            "reference_sphere_publish: tick=%d center=[%.3f %.3f %.3f] radius=%.3fm mode=%s radius_source=%s",
            int(self.planner_tick),
            float(reference_scene_center[0]),
            float(reference_scene_center[1]),
            float(reference_scene_center[2]),
            float(reference_radius),
            self.reference_center_mode,
            self.reference_radius_source,
        )

        # 记录端到端与策略内部耗时，便于在线性能诊断。
        planning_total_ms = (time.perf_counter() - planning_t0) * 1000.0
        mp_timing = getattr(self.motion_policy, "last_timing", {}) or {}
        self.profile_logger.info(
            "planning耗时：idx=%d total=%.2fms policy=%.2fms fisher=%.2fms "
            "history=%.2fms score=%.2fms gradient=%.2fms history_source=%s s2c=%.2fms model_v=%d",
            int(self.planner_tick),
            float(planning_total_ms),
            float(mp_timing.get("policy_total_ms", float("nan"))),
            float(mp_timing.get("fisher_ms", float("nan"))),
            float(mp_timing.get("history_ms", float("nan"))),
            float(mp_timing.get("score_ms", float("nan"))),
            float(mp_timing.get("gradient_ms", float("nan"))),
            str(mp_timing.get("history_source", "missing")),
            float(mp_timing.get("s2c_ms", float("nan"))),
            int(snapshot.model_version),
        )
        # 按是否 stop 归类统计，用于评估策略稳定性与有效动作比例。
        if bool(getattr(motion_result, "should_stop", False)):
            with self.stats_lock:
                self.stats.planner_stop_policy += 1
        else:
            with self.stats_lock:
                self.stats.planner_nonzero += 1
        self._maybe_log_status()


def build_argparser():
    parser = argparse.ArgumentParser(
        description="Distributed InfoFlow planning node (compute side).",
    )
    parser.add_argument(
        "-c", "--config", type=str, default="config/rtabmap_config.yaml"
    )
    parser.add_argument(
        "--camera_info_topic", type=str, default="/cam_1/color/camera_info"
    )
    parser.add_argument("--pose_topic", type=str, default="/omnimap/pose_state")
    parser.add_argument(
        "--snapshot_ref_topic", type=str, default="/omnimap/planner_snapshot_ref"
    )
    parser.add_argument(
        "--spherical_cmd_topic", type=str, default="/omnimap/spherical_cmd"
    )
    parser.add_argument("--world_frame", type=str, default="base_link")
    parser.add_argument("--planner_hz", type=float, default=None)
    parser.add_argument("--pose_stale_timeout_sec", type=float, default=None)
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=f"replica/output/{time.strftime('%Y%m%d_%H%M%S')}",
    )
    parser.add_argument("--fisher_step_scale", type=float, default=None)
    parser.add_argument("--linear_vel_max", type=float, default=None)
    parser.add_argument("--angular_gain", type=float, default=None)
    parser.add_argument("--radial_gain", type=float, default=None)
    parser.add_argument("--dt", type=float, default=None)
    parser.add_argument("--grad_eps", type=float, default=None)
    parser.add_argument("--spherical_speed_min", type=float, default=None)
    parser.add_argument(
        "--enable_angular", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--angular_speed_max", type=float, default=None)
    parser.add_argument(
        "--log_profile", choices=("quiet", "default", "debug"), default="default"
    )
    parser.add_argument("--log_level", type=str, default=None)
    parser.add_argument(
        "--log_section",
        action="append",
        choices=("all", "main", "tsdf", "gaussian", "fisher", "planner", "profile"),
        default=None,
    )
    parser.add_argument(
        "--log_min_level", choices=("DEBUG", "INFO", "WARNING"), default="INFO"
    )
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--status_log_interval_sec", type=float, default=1.0)
    parser.add_argument(
        "--log_file",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


if __name__ == "__main__":
    parser = build_argparser()
    args = parser.parse_args()
    set_nofile_limit()
    configure_entry_logging(args)
    config = load_runtime_config(args.config)
    mc = config["motion_control"]

    def _resolve(cli_val, yaml_val):
        return cli_val if cli_val is not None else yaml_val

    args.planner_hz = _resolve(args.planner_hz, mc["planner_hz"])
    args.pose_stale_timeout_sec = _resolve(
        args.pose_stale_timeout_sec, mc["pose_stale_timeout_sec"]
    )
    args.fisher_step_scale = _resolve(args.fisher_step_scale, mc["fisher_step_scale"])
    args.linear_vel_max = _resolve(args.linear_vel_max, mc["linear_vel_max"])
    args.angular_gain = _resolve(args.angular_gain, mc["angular_gain"])
    args.radial_gain = _resolve(args.radial_gain, mc["radial_gain"])
    args.dt = _resolve(args.dt, mc["dt"])
    args.grad_eps = _resolve(args.grad_eps, mc["grad_eps"])
    args.spherical_speed_min = _resolve(
        args.spherical_speed_min, mc["spherical_speed_min"]
    )
    args.enable_angular = _resolve(args.enable_angular, mc["enable_angular"])
    args.angular_speed_max = _resolve(args.angular_speed_max, mc["angular_speed_max"])

    node = InfoFlowPlanningNode(args, config)
    rospy.spin()
