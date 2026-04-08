import os  # nopep8

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import sys  # nopep8
from pathlib import Path  # nopep8

REPO_ROOT = Path(__file__).resolve().parent.parent
OMNIMAP_ROOT = REPO_ROOT / "omnimap"
for path in (REPO_ROOT, OMNIMAP_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import argparse
import resource
import time

import cv2
import numpy as np
import rospy
import tf2_ros
import torch
from cv_bridge import CvBridge
from geometry_msgs.msg import TransformStamped, TwistStamped
from message_filters import ApproximateTimeSynchronizer, Subscriber
from omni import OMNI
from omnimap.util.utils import load_config
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import CameraInfo, Image
from sim.motion_policy import FisherMotionPolicy
from tqdm import tqdm, trange

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (100000, rlimit[1]))


def save_trajectory(omni, all_inputs, output):
    """
    保存轨迹和图像为 Replica 数据集标准格式。
    """
    base_path = os.path.join(output, "imap", "00")
    rgb_out = os.path.join(base_path, "rgb")
    depth_out = os.path.join(base_path, "depth")
    os.makedirs(rgb_out, exist_ok=True)
    os.makedirs(depth_out, exist_ok=True)

    np.save(f"{output}/intrinsics.npy", omni.intrinsics.cpu().numpy())

    traj_full = []
    for i in trange(len(all_inputs), desc="Saving frames", unit="frame"):
        frame = all_inputs[i]

        rgb_image = frame["image"][0]
        rgb_image = rgb_image.transpose(1, 2, 0)
        rgb_image = cv2.cvtColor(rgb_image.astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(rgb_out, f"{i}.png"), rgb_image)

        depth_image = frame["depth"][0]
        depth_scale = frame["depth_scale"]
        depth_image = (depth_image * depth_scale).astype(np.uint16)
        cv2.imwrite(os.path.join(depth_out, f"{i}.png"), depth_image)

        pose_44 = frame["pose_44"][0]
        traj_full.append(pose_44.flatten())

    traj_full = np.stack(traj_full)
    np.savetxt(
        os.path.join(base_path, "traj_w_c.txt"),
        traj_full,
        fmt="%.18e",
        delimiter=" ",
    )

    print(f"Saved {len(traj_full)} frames to {base_path}")
    print(f"  - RGB images: {rgb_out}")
    print(f"  - Depth images: {depth_out}")
    print(f"  - Trajectory: {base_path}/traj_w_c.txt")


class InfoFlowROSNode:
    def __init__(self, args, config):
        rospy.init_node("info_flow_node", anonymous=True)

        rospy.loginfo("Waiting for camera info message to get intrinsics...")
        cam_info_msg = rospy.wait_for_message(args.camera_info_topic, CameraInfo)
        self.K = np.array(cam_info_msg.K, dtype=np.float64).reshape(3, 3)
        self.depth_scale = float(args.depth_scale)
        self.calib = np.array(
            [self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]],
            dtype=np.float32,
        )
        self.intrinsics = torch.tensor(self.calib.copy())

        self.bridge = CvBridge()
        self.index = 0
        self.progress_bar = tqdm(desc="InfoFlow")

        self.output = args.output
        self.all_inputs = []
        if self.output != "None":
            os.makedirs(self.output, exist_ok=True)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.world_frame = args.world_frame
        self.camera_frame = args.camera_frame
        self.cmd_frame = args.cmd_frame

        self.omni = OMNI(args, config)
        self.motion_policy = FisherMotionPolicy(
            step_gain_theta=args.fisher_step_scale,
            step_gain_phi=args.fisher_step_scale,
            cartesian=True,
            dt=args.dt,
            radial_gain=args.radial_gain,
            linear_vel_max=args.linear_vel_max,
            angular_gain=args.angular_gain,
            enable_angular=args.enable_angular,
            grad_eps=args.grad_eps,
            spherical_speed_min=args.spherical_speed_min,
            verbose=args.policy_verbose,
        )

        self.rgb_sub = Subscriber(args.rgb_topic, Image)
        self.depth_sub = Subscriber(args.depth_topic, Image)
        self.tsync = ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub], queue_size=10, slop=0.1
        )
        self.tsync.registerCallback(self.image_callback)

        self.cmd_pub = rospy.Publisher(args.cmd_topic, TwistStamped, queue_size=1)

        rospy.loginfo(
            "InfoFlowROSNode initialized, publishing TwistStamped commands to %s",
            args.cmd_topic,
        )

    def get_pose_from_tf(self, stamp):
        """
        从 TF 树获取相机 c2w 和 w2c pose 向量。
        """
        try:
            transform: TransformStamped = self.tf_buffer.lookup_transform(
                self.world_frame,
                self.camera_frame,
                stamp,
                rospy.Duration(0.1),
            )

            t = transform.transform.translation
            q = transform.transform.rotation

            c2w = np.eye(4, dtype=np.float64)
            c2w[:3, 3] = [t.x, t.y, t.z]
            c2w[:3, :3] = R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()

            w2c = np.linalg.inv(c2w)
            quat = R.from_matrix(w2c[:3, :3]).as_quat()
            pose = np.hstack((w2c[:3, 3], quat))

            return pose, c2w
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as exc:
            rospy.logwarn("TF lookup failed: %s", exc)
            return None, None

    def publish_zero_twist(self, stamp=None):
        msg = TwistStamped()
        msg.header.stamp = stamp if stamp is not None else rospy.Time.now()
        msg.header.frame_id = self.cmd_frame
        self.cmd_pub.publish(msg)

    def publish_motion_result(self, motion_result, stamp=None):
        msg = TwistStamped()
        msg.header.stamp = stamp if stamp is not None else rospy.Time.now()
        msg.header.frame_id = self.cmd_frame

        linear = np.asarray(motion_result.velocity_world, dtype=np.float64).reshape(3)
        angular = np.asarray(
            motion_result.angular_velocity_world, dtype=np.float64
        ).reshape(3)

        msg.twist.linear.x = float(linear[0])
        msg.twist.linear.y = float(linear[1])
        msg.twist.linear.z = float(linear[2])
        msg.twist.angular.x = float(angular[0])
        msg.twist.angular.y = float(angular[1])
        msg.twist.angular.z = float(angular[2])
        self.cmd_pub.publish(msg)

    def maybe_publish_velocity(self, pose_4x4, image_shape, stamp):
        try:
            motion_result = self.motion_policy.next_pose_from_c2w(
                gs_backend=self.omni.gs,
                current_c2w=np.asarray(pose_4x4, dtype=np.float64),
                intrinsics_vec=self.calib,
                image_size=image_shape,
                idx=self.index,
            )
        except Exception as exc:
            rospy.logwarn_throttle(
                2.0,
                "Fisher policy not ready, publishing zero twist: %s",
                exc,
            )
            self.publish_zero_twist(stamp)
            return

        if motion_result.should_stop:
            self.publish_zero_twist(stamp)
            return

        self.publish_motion_result(motion_result, stamp)

    def image_callback(self, rgb_msg, depth_msg):
        pose, pose_4x4 = self.get_pose_from_tf(rgb_msg.header.stamp)
        if pose is None:
            self.publish_zero_twist(rgb_msg.header.stamp)
            return

        try:
            rgb_image = self.bridge.imgmsg_to_cv2(rgb_msg, "rgb8")
            depth_image = self.bridge.imgmsg_to_cv2(
                depth_msg, desired_encoding="passthrough"
            )
        except Exception as exc:
            rospy.logerr("CV Bridge error: %s", exc)
            self.publish_zero_twist(rgb_msg.header.stamp)
            return

        image = torch.as_tensor(rgb_image).permute(2, 0, 1)
        depth = torch.as_tensor(depth_image.astype(np.float32) / self.depth_scale)
        pose_tensor = torch.as_tensor(pose)
        pose_4x4_tensor = torch.as_tensor(pose_4x4)

        if self.output != "None":
            frame_data = {
                "index": self.index,
                "image": image[None].cpu().numpy(),
                "depth": depth[None].cpu().numpy(),
                "pose": pose_tensor[None].cpu().numpy(),
                "intrinsics": self.intrinsics[None].cpu().numpy(),
                "pose_44": pose_4x4_tensor[None].cpu().numpy(),
                "is_last": False,
                "depth_scale": self.depth_scale,
            }
            self.all_inputs.append(frame_data)

        try:
            self.omni.track(
                self.index,
                image[None],
                depth[None],
                pose_tensor[None],
                self.progress_bar,
                intrinsics=self.intrinsics[None],
                is_last=False,
                pose_44=pose_4x4_tensor[None],
            )
        except Exception as exc:
            rospy.logerr("OMNI.track failed: %s", exc)
            self.publish_zero_twist(rgb_msg.header.stamp)
            return
        self.maybe_publish_velocity(
            pose_4x4=pose_4x4,
            image_shape=(rgb_image.shape[0], rgb_image.shape[1]),
            stamp=rgb_msg.header.stamp,
        )
        self.index += 1

    def terminate(self):
        self.publish_zero_twist()
        if self.output != "None":
            save_trajectory(self.omni, self.all_inputs, self.output)
        self.progress_bar.close()
        self.omni.terminate()
        rospy.signal_shutdown("InfoFlow ROS Node terminated")


def build_argparser():
    parser = argparse.ArgumentParser(
        description=(
            "InfoFlow ROS Node\n"
            "订阅 RGBD + TF，内部运行 OmniMap 与 FisherMotionPolicy，\n"
            "并向 /servo_server/delta_twist_cmds 发布 TwistStamped 速度命令。\n"
            "\nUsage:\n"
            "source source_env.sh\n"
            "python info_flow/info_flow_node.py \\\n"
            "  --config config/rtabmap_config.yaml \\\n"
            "  --rgb_topic /cam_1/color/image_raw \\\n"
            "  --depth_topic /cam_1/aligned_depth_to_color/image_raw \\\n"
            "  --camera_info_topic /cam_1/color/camera_info \\\n"
            "  --world_frame base_link \\\n"
            "  --camera_frame cam_1_color_optical_frame\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="config/rtabmap_config.yaml",
        help="config file path",
    )
    parser.add_argument(
        "-r",
        "--rgb_topic",
        type=str,
        default="/cam_1/color/image_raw",
        help="RGB image topic",
    )
    parser.add_argument(
        "-d",
        "--depth_topic",
        type=str,
        default="/cam_1/aligned_depth_to_color/image_raw",
        help="Depth image topic",
    )
    parser.add_argument(
        "--camera_info_topic",
        type=str,
        default="/cam_1/color/camera_info",
        help="camera info topic",
    )
    parser.add_argument(
        "--world_frame",
        type=str,
        default="base_link",
        help="world frame id",
    )
    parser.add_argument(
        "--camera_frame",
        type=str,
        default="cam_1_color_optical_frame",
        help="camera frame id",
    )
    parser.add_argument(
        "--cmd_topic",
        type=str,
        default="/servo_server/delta_twist_cmds",
        help="output TwistStamped topic",
    )
    parser.add_argument(
        "--cmd_frame",
        type=str,
        default="base_link",
        help="frame_id used for TwistStamped commands",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=f"replica/output/{time.strftime('%Y%m%d_%H%M%S')}",
        help="output path",
    )
    parser.add_argument(
        "--depth_scale",
        type=float,
        default=1000.0,
        help="depth scale factor",
    )
    parser.add_argument(
        "--fisher_step_scale",
        type=float,
        default=1e-4,
        help="shared Fisher control scale for theta/phi",
    )
    parser.add_argument(
        "--linear_vel_max",
        type=float,
        default=0.5,
        help="maximum Cartesian linear velocity",
    )
    parser.add_argument(
        "--angular_gain",
        type=float,
        default=2.0,
        help="angular gain for omega command",
    )
    parser.add_argument(
        "--radial_gain",
        type=float,
        default=0.2,
        help="radial correction gain",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.1,
        help="control timestep used by FisherMotionPolicy",
    )
    parser.add_argument(
        "--grad_eps",
        type=float,
        default=0.01,
        help="finite difference epsilon for Fisher gradients",
    )
    parser.add_argument(
        "--spherical_speed_min",
        type=float,
        default=1e-4,
        help="minimum spherical speed before publishing zero twist",
    )
    parser.add_argument(
        "--enable_angular",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enable angular velocity output",
    )
    parser.add_argument(
        "--policy_verbose",
        action="store_true",
        help="enable verbose FisherMotionPolicy logging",
    )
    parser.add_argument(
        "--vis_gui",
        action="store_true",
        help="use opencv to visuliazation the whole process",
    )
    return parser


if __name__ == "__main__":
    parser = build_argparser()
    args = parser.parse_args()

    config = load_config(args.config)
    try:
        torch.multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass

    node = InfoFlowROSNode(args, config)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down...")
    finally:
        node.terminate()
        print(f"Done, results saved to {node.output}")
