from __future__ import annotations

import argparse
import threading
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
import rospy
import tf2_ros
import torch
from geometry_msgs.msg import PoseStamped
from message_filters import ApproximateTimeSynchronizer, Subscriber
from sensor_msgs.msg import CompressedImage
from tqdm import tqdm

from distributed_common import (
    SnapshotStore,
    configure_entry_logging,
    import_omnimap_msgs,
    load_runtime_config,
    lookup_pose_from_tf,
    make_run_id,
    pose_stamped_from_c2w,
    set_nofile_limit,
    wait_for_camera_calibration,
)
from omni import OMNI
from omnimap.util.utils import get_section_logger, should_log_step
from planner_snapshot import build_planner_snapshot
from slam_frontend import DropOldestQueue, RTabMapKeyframeGate


@dataclass
class TrackTask:
    index: int
    stamp: rospy.Time
    enqueue_wall_time: float
    pose_w2c: np.ndarray
    pose_4x4: np.ndarray
    image: torch.Tensor
    depth: torch.Tensor
    image_shape: Tuple[int, int]
    depth_valid_ratio: float
    source: str


@dataclass
class TrackingStats:
    pose_published: int = 0
    frame_candidates: int = 0
    gated_passed: int = 0
    gated_by_interval: int = 0
    gated_by_motion: int = 0
    forced_gap_keyframes: int = 0
    frames_enqueued: int = 0
    queue_dropped: int = 0
    track_success: int = 0
    track_fail: int = 0
    snapshot_success: int = 0
    snapshot_fail: int = 0


class InfoFlowTrackingNode:
    def __init__(self, args, config):
        rospy.init_node("info_flow_tracking_node", anonymous=True)
        PlannerSnapshotRef, _ = import_omnimap_msgs()

        self.args = args
        self.config = config
        self.main_logger = get_section_logger("entry.infoflow_tracking", "main")
        self.profile_logger = get_section_logger("profile.infoflow_tracking", "profile")
        self.world_frame = str(args.world_frame)
        self.camera_frame = str(args.camera_frame)
        self.depth_scale = float(args.depth_scale)
        self.log_every = max(1, int(args.log_every))
        self.status_log_interval_sec = max(0.2, float(args.status_log_interval_sec))
        self.max_frames = int(args.max_frames)
        self.force_keyframe_gap_frames = 30
        self.run_id = str(args.run_id or make_run_id("infoflow"))
        self.progress_bar = tqdm(
            desc="InfoFlowTracking",
            disable=(str(args.log_profile).lower() == "quiet"),
        )

        self.main_logger.info("正在等待相机内参消息...")
        _, K, calib, image_size_hw = wait_for_camera_calibration(args.camera_info_topic)
        self.K = K
        self.calib = calib
        self.intrinsics = torch.tensor(self.calib.copy())
        self.image_size_hw = image_size_hw

        self.pose_pub = rospy.Publisher(args.pose_topic, PoseStamped, queue_size=1)
        self.snapshot_ref_pub = rospy.Publisher(
            args.snapshot_ref_topic,
            PlannerSnapshotRef,
            queue_size=1,
        )

        self.omni = OMNI(args, config)
        self.snapshot_store = SnapshotStore(
            root_dir=str(args.snapshot_store_dir),
            run_id=self.run_id,
            retention=int(args.snapshot_retention),
            logger=self.main_logger,
        )

        self.keyframe_gate = RTabMapKeyframeGate(
            min_interval_sec=float(args.keyframe_min_interval_sec),
            min_translation_m=float(args.keyframe_min_translation_m),
            min_rotation_deg=float(args.keyframe_min_rotation_deg),
            forced_gap_frames=int(self.force_keyframe_gap_frames),
        )
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.track_queue = DropOldestQueue(maxsize=int(args.track_queue_size))
        self.track_stop_event = threading.Event()
        self.track_worker_thread = threading.Thread(
            target=self._tracking_worker,
            name="infoflow_tracking_worker",
            daemon=True,
        )
        self.track_worker_thread.start()

        self.rgb_sub = Subscriber(args.rgb_topic, CompressedImage)
        self.depth_sub = Subscriber(args.depth_topic, CompressedImage)
        self.tsync = ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub],
            queue_size=30,
            slop=max(0.0, float(args.sync_slop_sec)),
        )
        self.tsync.registerCallback(self.tf_native_callback)

        self.stats_lock = threading.Lock()
        self.stats = TrackingStats()
        self.next_track_index = 0
        self.model_version = 0
        self.shutdown_requested = False
        self._last_status_wall = float(time.monotonic())
        self._last_status_track_ok = 0
        self._last_status_snapshot_ok = 0
        self._last_status_pose_pub = 0
        self._last_queue_drop_log_wall = 0.0
        self._throttle_last = {}

        self.main_logger.info(
            "Tracking 节点已启动：pose_topic=%s snapshot_ref_topic=%s run_id=%s",
            args.pose_topic,
            args.snapshot_ref_topic,
            self.run_id,
        )
        rospy.on_shutdown(self.terminate)

    def _tracked_frame_limit_reached(self) -> bool:
        with self.stats_lock:
            return int(self.stats.track_success) >= int(self.max_frames)

    def _decode_compressed_rgbd(self, rgb_msg, depth_msg):
        rgb_buffer = np.frombuffer(rgb_msg.data, dtype=np.uint8)
        rgb_bgr = cv2.imdecode(rgb_buffer, cv2.IMREAD_COLOR)
        if rgb_bgr is None:
            raise ValueError("failed to decode compressed RGB image")
        rgb_image = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)

        depth_buffer = np.frombuffer(depth_msg.data, dtype=np.uint8)
        depth_image = cv2.imdecode(depth_buffer, cv2.IMREAD_UNCHANGED)
        if depth_image is None:
            raise ValueError("failed to decode compressed depth image")
        if depth_image.ndim == 3:
            depth_image = cv2.cvtColor(depth_image, cv2.COLOR_BGR2GRAY)
        return rgb_image, depth_image

    def _build_track_task(
        self,
        *,
        idx: int,
        stamp: rospy.Time,
        pose_w2c: np.ndarray,
        pose_4x4: np.ndarray,
        rgb_image: np.ndarray,
        depth_image: np.ndarray,
        source: str,
    ) -> TrackTask:
        image = torch.as_tensor(rgb_image).permute(2, 0, 1)
        depth = torch.as_tensor(depth_image.astype(np.float32) / self.depth_scale)
        return TrackTask(
            index=int(idx),
            stamp=stamp,
            enqueue_wall_time=float(time.perf_counter()),
            pose_w2c=np.asarray(pose_w2c, dtype=np.float64),
            pose_4x4=np.asarray(pose_4x4, dtype=np.float64),
            image=image,
            depth=depth,
            image_shape=(int(rgb_image.shape[0]), int(rgb_image.shape[1])),
            depth_valid_ratio=float(np.mean(depth_image > 0)),
            source=str(source),
        )

    def _enqueue_track_task(self, task: TrackTask) -> None:
        dropped = self.track_queue.put(task)
        with self.stats_lock:
            self.stats.frames_enqueued += 1
            if dropped is not None:
                self.stats.queue_dropped += 1
                dropped_total = int(self.stats.queue_dropped)
            else:
                dropped_total = int(self.stats.queue_dropped)
        if dropped is not None:
            now = float(time.monotonic())
            if (now - self._last_queue_drop_log_wall) >= self.status_log_interval_sec:
                self._last_queue_drop_log_wall = now
                self.main_logger.debug(
                    "跟踪队列积压：丢弃最旧帧 idx=%d，新帧 idx=%d，队列=%d，累计丢弃=%d",
                    int(dropped.index),
                    int(task.index),
                    int(self.track_queue.qsize()),
                    dropped_total,
                )

    def _log_throttle(self, key: str, interval_sec: float, level: str, msg: str, *args):
        now = float(time.monotonic())
        last = float(self._throttle_last.get(key, -1e18))
        if (now - last) < float(interval_sec):
            return
        self._throttle_last[key] = now
        getattr(self.main_logger, str(level).lower())(msg, *args)

    def _maybe_log_status(self) -> None:
        now = float(time.monotonic())
        elapsed = now - self._last_status_wall
        if elapsed < self.status_log_interval_sec:
            return
        with self.stats_lock:
            stats = TrackingStats(**self.stats.__dict__)
        d_track_ok = int(stats.track_success) - int(self._last_status_track_ok)
        d_snapshot_ok = int(stats.snapshot_success) - int(self._last_status_snapshot_ok)
        d_pose_pub = int(stats.pose_published) - int(self._last_status_pose_pub)
        self._last_status_wall = now
        self._last_status_track_ok = int(stats.track_success)
        self._last_status_snapshot_ok = int(stats.snapshot_success)
        self._last_status_pose_pub = int(stats.pose_published)
        self.main_logger.info(
            (
                "Tracking 状态：pose_hz=%.1f track_hz=%.1f snapshot_hz=%.1f "
                "queue=%d/%d 累计(pose=%d candidates=%d gate_passed=%d enqueued=%d dropped=%d "
                "track_ok=%d track_fail=%d snapshot_ok=%d snapshot_fail=%d)"
            ),
            float(d_pose_pub / max(elapsed, 1e-6)),
            float(d_track_ok / max(elapsed, 1e-6)),
            float(d_snapshot_ok / max(elapsed, 1e-6)),
            int(self.track_queue.qsize()),
            int(self.track_queue.maxsize),
            int(stats.pose_published),
            int(stats.frame_candidates),
            int(stats.gated_passed),
            int(stats.frames_enqueued),
            int(stats.queue_dropped),
            int(stats.track_success),
            int(stats.track_fail),
            int(stats.snapshot_success),
            int(stats.snapshot_fail),
        )

    def tf_native_callback(self, rgb_msg, depth_msg):
        # RGB-D 同步回调（原生 TF 路径）：
        # 1) 用图像时间戳查询相机位姿；
        # 2) 进行关键帧门控；
        # 3) 通过门控后解码图像并入队给 tracking 后端。
        if self.shutdown_requested:
            return
        stamp = rgb_msg.header.stamp
        stamp_sec = float(stamp.to_sec()) if stamp is not None else 0.0
        now_sec = float(rospy.Time.now().to_sec())
        self.profile_logger.debug(
            "sync_cb: stamp=%.3f ros_lag=%.3fs",
            float(stamp_sec),
            float(now_sec - stamp_sec),
        )
        try:
            # 按图像时间戳查询 world->camera 位姿，保证时序对齐。
            tf_t0 = time.perf_counter()
            pose_w2c, pose_4x4, _ = lookup_pose_from_tf(
                self.tf_buffer,
                world_frame=self.world_frame,
                camera_frame=self.camera_frame,
                stamp=stamp,
            )
            self.profile_logger.debug(
                "tf_lookup: stamp=%.3f tf_ms=%.2f success=True",
                float(stamp_sec),
                float((time.perf_counter() - tf_t0) * 1000.0),
            )
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as exc:
            self.profile_logger.debug("tf_lookup: stamp=%.3f success=False", float(stamp_sec))
            self._log_throttle(
                "tf_lookup_failure",
                1.0,
                "warning",
                # 节流告警：避免 TF 问题在高帧率下刷屏。
                "按图像时间戳查询 TF 失败：rgb_stamp=%.3f now=%.3f lag=%.3fs err=%s",
                float(stamp_sec),
                float(now_sec),
                float(now_sec - stamp_sec),
                exc,
            )
            return

        # 发布当前帧位姿供外部可视化/监控，并更新统计。
        pose_msg = pose_stamped_from_c2w(
            c2w=pose_4x4,
            stamp=stamp,
            frame_id=self.world_frame,
        )
        self.pose_pub.publish(pose_msg)
        with self.stats_lock:
            self.stats.pose_published += 1
            self.stats.frame_candidates += 1

        if self._tracked_frame_limit_reached():
            self.request_shutdown(f"达到跟踪帧上限（{self.max_frames}），停止 tracking 节点。")
            return

        # 关键帧门控：根据时间间隔、位姿运动量等条件决定是否送入 tracking。
        decision = self.keyframe_gate.decide(
            pose_4x4=pose_4x4,
            stamp_sec=stamp_sec,
            frame_index=int(self.next_track_index),
        )
        if decision.reason == "forced_gap_keyframe":
            with self.stats_lock:
                self.stats.forced_gap_keyframes += 1
            self.main_logger.info(
                "关键帧门控连续拒绝达 %d 帧，已强制当前帧作为 keyframe 入队。",
                int(self.force_keyframe_gap_frames),
            )
        if not decision.should_track:
            with self.stats_lock:
                if decision.reason == "below_min_interval":
                    self.stats.gated_by_interval += 1
                elif decision.reason == "below_motion_threshold":
                    self.stats.gated_by_motion += 1
            self._maybe_log_status()
            return

        with self.stats_lock:
            self.stats.gated_passed += 1

        try:
            # 仅对通过门控的帧做解码，降低无效 CPU 开销。
            rgb_image, depth_image = self._decode_compressed_rgbd(rgb_msg, depth_msg)
        except Exception as exc:
            self.main_logger.error("压缩图像解码失败：%s", exc)
            return

        # 构造 tracking 任务并入队，source 记录触发关键帧的原因，便于回溯分析。
        task = self._build_track_task(
            idx=self.next_track_index,
            stamp=stamp,
            pose_w2c=pose_w2c,
            pose_4x4=pose_4x4,
            rgb_image=rgb_image,
            depth_image=depth_image,
            source=f"tf_keyframe:{decision.reason}",
        )
        self.next_track_index += 1
        self._enqueue_track_task(task)
        if should_log_step(task.index, self.log_every):
            self.main_logger.debug(
                "已入队关键帧：frame=%d source=%s queue=%d",
                int(task.index),
                task.source,
                int(self.track_queue.qsize()),
            )
        self._maybe_log_status()

    def _publish_snapshot_ref(self, *, snapshot, snapshot_path: str, stamp: rospy.Time) -> None:
        PlannerSnapshotRef, _ = import_omnimap_msgs()
        msg = PlannerSnapshotRef()
        msg.header.stamp = stamp
        msg.header.frame_id = self.world_frame
        msg.run_id = self.run_id
        msg.model_version = int(snapshot.model_version)
        msg.keyframe_idx = int(snapshot.keyframe_idx)
        msg.snapshot_uri = str(snapshot_path)
        msg.created_wall_time = float(snapshot.created_wall_time)
        msg.runtime_device_hint = str(snapshot.backend.get_runtime_device())
        self.snapshot_ref_pub.publish(msg)

    def _tracking_worker(self):
        while not self.track_stop_event.is_set() or not self.track_queue.empty():
            try:
                task = self.track_queue.get(timeout=0.1)
            except TimeoutError:
                continue

            pose_tensor = torch.as_tensor(task.pose_w2c)
            pose_4x4_tensor = torch.as_tensor(task.pose_4x4)
            worker_t0 = time.perf_counter()
            queue_wait_ms = float((worker_t0 - task.enqueue_wall_time) * 1000.0)
            try:
                track_t0 = time.perf_counter()
                self.omni.track(
                    task.index,
                    task.image[None],
                    task.depth[None],
                    pose_tensor[None],
                    self.progress_bar,
                    intrinsics=self.intrinsics[None],
                    is_last=False,
                    pose_44=pose_4x4_tensor[None],
                    update_rate=self.log_every,
                )
                track_ms = float((time.perf_counter() - track_t0) * 1000.0)
            except Exception as exc:
                self.main_logger.error("跟踪阶段 OMNI.track 执行失败：%s", exc)
                with self.stats_lock:
                    self.stats.track_fail += 1
                self._maybe_log_status()
                continue

            with self.stats_lock:
                next_version = int(self.model_version) + 1
            snapshot_ms = 0.0
            try:
                snapshot = build_planner_snapshot(
                    live_backend=self.omni.gs,
                    model_version=next_version,
                    keyframe_idx=task.index,
                )
                save_t0 = time.perf_counter()
                snapshot_path = self.snapshot_store.save(snapshot)
                snapshot_ms = float((time.perf_counter() - save_t0) * 1000.0)
                self.profile_logger.info(
                    "snapshot_save: model_v=%d frame=%d save_ms=%.2f path=%s",
                    int(snapshot.model_version),
                    int(task.index),
                    snapshot_ms,
                    str(snapshot_path),
                )
                with self.stats_lock:
                    self.model_version = int(snapshot.model_version)
                    self.stats.snapshot_success += 1
                self._publish_snapshot_ref(
                    snapshot=snapshot,
                    snapshot_path=str(snapshot_path),
                    stamp=task.stamp,
                )
            except Exception as exc:
                self.main_logger.error("构建或落盘规划快照失败：%s", exc)
                with self.stats_lock:
                    self.stats.snapshot_fail += 1

            with self.stats_lock:
                self.stats.track_success += 1
                tracked_ok = int(self.stats.track_success)

            if should_log_step(task.index, self.log_every):
                total_worker_ms = float((time.perf_counter() - worker_t0) * 1000.0)
                self.profile_logger.info(
                    (
                        "tracking_timing: frame=%d queue_wait_ms=%.2f "
                        "track_ms=%.2f snapshot_ms=%.2f total_worker_ms=%.2f "
                        "queue=%d/%d dropped=%d over_1hz=%s"
                    ),
                    int(task.index),
                    queue_wait_ms,
                    track_ms,
                    snapshot_ms,
                    total_worker_ms,
                    int(self.track_queue.qsize()),
                    int(self.track_queue.maxsize),
                    int(self.stats.queue_dropped),
                    str(bool(track_ms > 1000.0 or total_worker_ms > 1000.0)),
                )
                self.main_logger.info(
                    "跟踪状态：frame=%d source=%s depth_valid_ratio=%.3f model_v=%d queue=%d/%d dropped=%d",
                    int(task.index),
                    task.source,
                    float(task.depth_valid_ratio),
                    int(self.model_version),
                    int(self.track_queue.qsize()),
                    int(self.track_queue.maxsize),
                    int(self.stats.queue_dropped),
                )
            self._maybe_log_status()

            if tracked_ok >= self.max_frames:
                self.request_shutdown(f"达到跟踪帧上限（{self.max_frames}），停止 tracking 节点。")

    def request_shutdown(self, reason: str) -> None:
        if self.shutdown_requested:
            return
        self.shutdown_requested = True
        self.main_logger.info("%s", reason)
        self.track_stop_event.set()
        self.track_queue.close()
        rospy.signal_shutdown(reason)

    def terminate(self) -> None:
        self.track_stop_event.set()
        self.track_queue.close()
        if self.track_worker_thread.is_alive():
            self.track_worker_thread.join(timeout=3.0)
        self.progress_bar.close()


def build_argparser():
    parser = argparse.ArgumentParser(
        description="Distributed InfoFlow tracking node (compute side).",
    )
    parser.add_argument("-c", "--config", type=str, default="config/rtabmap_config.yaml")
    parser.add_argument("-r", "--rgb_topic", type=str, default="/cam_1/color/image_raw/compressed")
    parser.add_argument("-d", "--depth_topic", type=str, default="/cam_1/aligned_depth_to_color/image_raw/compressed")
    parser.add_argument("--camera_info_topic", type=str, default="/cam_1/color/camera_info")
    parser.add_argument("--world_frame", type=str, default="base_link")
    parser.add_argument("--camera_frame", type=str, default="cam_1_color_optical_frame")
    parser.add_argument("--sync_slop_sec", type=float, default=0.12)
    parser.add_argument("--keyframe_min_interval_sec", type=float, default=0.10)
    parser.add_argument("--keyframe_min_translation_m", type=float, default=0.01)
    parser.add_argument("--keyframe_min_rotation_deg", type=float, default=1.0)
    parser.add_argument("--track_queue_size", type=int, default=2)
    parser.add_argument("--max_frames", type=int, default=500)
    parser.add_argument("--pose_topic", type=str, default="/omnimap/pose_state")
    parser.add_argument("--snapshot_ref_topic", type=str, default="/omnimap/planner_snapshot_ref")
    parser.add_argument("--snapshot_store_dir", type=str, default="/tmp/omnimap_snapshots")
    parser.add_argument("--snapshot_retention", type=int, default=4)
    parser.add_argument("--run_id", type=str, default="")
    parser.add_argument("-o", "--output", type=str, default=f"replica/output/{time.strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--depth_scale", type=float, default=1000.0)
    parser.add_argument("--log_profile", choices=("quiet", "default", "debug"), default="default")
    parser.add_argument("--log_level", type=str, default=None)
    parser.add_argument(
        "--log_section",
        action="append",
        choices=("all", "main", "tsdf", "gaussian", "fisher", "planner", "profile"),
        default=None,
    )
    parser.add_argument("--log_min_level", choices=("DEBUG", "INFO", "WARNING"), default="INFO")
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--status_log_interval_sec", type=float, default=1.0)
    parser.add_argument(
        "--log_file",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--vis_gui", action="store_true")
    return parser


if __name__ == "__main__":
    parser = build_argparser()
    args = parser.parse_args()
    set_nofile_limit()
    configure_entry_logging(args)
    config = load_runtime_config(args.config)
    try:
        torch.multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass
    node = InfoFlowTrackingNode(args, config)
    rospy.spin()
