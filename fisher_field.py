import os  # nopep8

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import sys  # nopep8
sys.path.append(os.path.join(os.path.dirname(__file__), "omnimap"))  # nopep8

import time
import torch
import cv2
import re
import os
import argparse
import numpy as np
import lietorch
import resource

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (100000, rlimit[1]))

from omnimap.util.utils import load_config
from tqdm import tqdm, trange
from torch.multiprocessing import Process, Queue
from omni import OMNI
from natsort import natsorted
from scipy.spatial.transform import Rotation as R

import rospy
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from message_filters import ApproximateTimeSynchronizer, Subscriber
from cv_bridge import CvBridge
import tf2_ros
from geometry_msgs.msg import TransformStamped

def save_trajectory(omni, all_inputs, output):
    """
    保存轨迹和图像为 Replica 数据集标准格式
    结构:
      output/
        imap/00/
          rgb/         - 0.png, 1.png, ...
          depth/       - 0.png, 1.png, ...
          traj_w_c.txt - 每行 16 个数值 (4x4 c2w 矩阵展平)
        intrinsics.npy
    """
    # 创建目录结构
    base_path = os.path.join(output, "imap", "00")
    rgb_out = os.path.join(base_path, "rgb")
    depth_out = os.path.join(base_path, "depth")
    os.makedirs(rgb_out, exist_ok=True)
    os.makedirs(depth_out, exist_ok=True)
    
    # 保存内参
    np.save(f"{output}/intrinsics.npy", omni.intrinsics.cpu().numpy())
    
    # 保存轨迹和图像
    traj_full = []
    for i in trange(len(all_inputs), desc="Saving frames", unit="frame"):
        frame = all_inputs[i]
        
        # 1. 保存 RGB 图像 (转换回 BGR 格式保存)
        rgb_image = frame["image"][0]  # [C, H, W]
        rgb_image = rgb_image.transpose(1, 2, 0)  # [H, W, C]
        rgb_image = cv2.cvtColor(rgb_image.astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(rgb_out, f"{i}.png"), rgb_image)
        
        # 2. 保存 Depth 图像 (转换回原始深度值，保存为 16-bit PNG)
        depth_image = frame["depth"][0]  # [H, W]
        depth_scale = frame["depth_scale"]
        depth_image = (depth_image * depth_scale).astype(np.uint16)
        cv2.imwrite(os.path.join(depth_out, f"{i}.png"), depth_image)
        
        # 3. 保存位姿 (c2w 4x4 矩阵展平为 16 列)
        pose_44 = frame["pose_44"][0]  # [4, 4]
        traj_full.append(pose_44.flatten())
    
    traj_full = np.stack(traj_full)
    np.savetxt(os.path.join(base_path, "traj_w_c.txt"), traj_full, fmt='%.18e', delimiter=' ')
    
    print(f"Saved {len(traj_full)} frames to {base_path}")
    print(f"  - RGB images: {rgb_out}")
    print(f"  - Depth images: {depth_out}")
    print(f"  - Trajectory: {base_path}/traj_w_c.txt")

def to_se3_matrix(pvec):
    pose = np.eye(4)
    pose[:3, :3] = R.from_quat(pvec[4:]).as_matrix()
    pose[:3, 3] = pvec[1:4]
    return pose

class FisherFieldPlanner:
    def __init__(self, args, config):
        rospy.init_node('omni_ros_node', anonymous=True)

        # 订阅 CameraInfo 获取相机内参
        rospy.loginfo("Waiting for camera info message to get intrinsics...")
        cam_info_msg = rospy.wait_for_message(args.camera_info_topic, CameraInfo)
        self.K = np.array(cam_info_msg.K).reshape(3, 3)
        self.depth_scale = args.depth_scale # 确保这里是 1000.0 (mm->m)
        self.calib = np.array([self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]])
        self.intrinsics = torch.tensor(self.calib[:4].copy())
        
        self.bridge = CvBridge()
        self.index = 0
        self.progress_bar = tqdm(desc="Training")

        # 输出路径
        self.output = args.output
        self.all_inputs = []
        if self.output != "None":
            os.makedirs(self.output, exist_ok=True)

        # TF2 监听器
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.world_frame = args.world_frame
        self.camera_frame = args.camera_frame
        
        # 初始化 OMNI
        self.omni = OMNI(args, config)
        
        # 同步订阅RGB和Depth图像
        self.rgb_sub = Subscriber(args.rgb_topic, Image)
        self.depth_sub = Subscriber(args.depth_topic, Image)
        self.tsync = ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub], queue_size=10, slop=0.1
        )
        self.tsync.registerCallback(self.image_callback)
        self.target_pose_pub = rospy.Publisher("/omnimap/target_pose", PoseStamped, queue_size=1)
        self.target_frame_id = "world"
        
        rospy.loginfo("OmniROSNode initialized, waiting for images and TF...")


    def get_pose_from_tf(self, stamp):
        """
        从 TF 树获取相机位姿
        返回: pose (7维向量), pose_4x4 (4x4矩阵), 或 (None, None) 如果失败
        """
        try:
            # 查询 world_frame -> camera_frame 的变换 (即 c2w)
            transform: TransformStamped = self.tf_buffer.lookup_transform(
                self.world_frame,   # target frame (world)
                self.camera_frame,  # source frame (camera)
                stamp,
                rospy.Duration(0.1)  # 等待超时
            )
            
            # 提取平移和旋转
            t = transform.transform.translation
            q = transform.transform.rotation
            
            # 构建 c2w (camera-to-world) 4x4 矩阵
            c2w = np.eye(4)
            c2w[:3, 3] = [t.x, t.y, t.z]
            c2w[:3, :3] = R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
            
            # 计算 w2c (world-to-camera) 用于 pose 向量
            w2c = np.linalg.inv(c2w)
            quat = R.from_matrix(w2c[:3, :3]).as_quat()  # [qx, qy, qz, qw]
            pose = np.hstack((w2c[:3, 3], quat))  # [tx, ty, tz, qx, qy, qz, qw]
            
            return pose, c2w
            
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, 
                tf2_ros.ExtrapolationException) as e:
            rospy.logwarn(f"TF lookup failed: {e}")
            return None, None

    def image_callback(self, rgb_msg, depth_msg):
        """处理同步的 RGB 和深度图像"""
        # 获取时间戳对应的位姿
        pose, pose_4x4 = self.get_pose_from_tf(rgb_msg.header.stamp)
        if pose is None:
            rospy.logwarn("Skipping frame due to TF lookup failure")
            return
        
        # 转换图像
        try:
            rgb_image = self.bridge.imgmsg_to_cv2(rgb_msg, "rgb8")
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        except Exception as e:
            rospy.logerr(f"CV Bridge error: {e}")
            return
        
        # 转换为 tensor
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
        self.publish_target_from_gs()
        self.index += 1

    def publish_target_from_gs(self):
        if not hasattr(self.omni.gs, "next_viewpoint"):
            return
        vp = self.omni.gs.next_viewpoint
        if vp is None:
            return

        # c2w = (world_view_transform.T).inverse()
        c2w = (vp.world_view_transform.T).inverse().cpu().numpy()
        pos = c2w[:3, 3]
        quat = R.from_matrix(c2w[:3, :3]).as_quat()  # [x, y, z, w]

        msg = PoseStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = self.target_frame_id
        msg.pose.position.x = float(pos[0])
        msg.pose.position.y = float(pos[1])
        msg.pose.position.z = float(pos[2])
        msg.pose.orientation.x = float(quat[0])
        msg.pose.orientation.y = float(quat[1])
        msg.pose.orientation.z = float(quat[2])
        msg.pose.orientation.w = float(quat[3])

        self.target_pose_pub.publish(msg)
        
    def terminate(self):
        if self.output != "None":
            save_trajectory(self.omni, self.all_inputs, self.output)
        self.progress_bar.close()
        self.omni.terminate()
        # 关闭ros节点
        rospy.signal_shutdown("OmniMap ROS Node terminated")
        
     
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "OmniMap ROS Node\n"
            "如果需要保存输出，请添加 --save 参数，\n"
            "默认输出路径为 replica/output/YYYYMMDD_HHMMSS\n"
            "\nUsage:\n"
            "source source_env.sh \n"
            "python omni_ros_node.py \\\n"
            "--config config/rtabmap_config.yaml \\\n"
            "--rgb_topic /camera/color/image_raw \\\n"
            "--depth_topic /camera/aligned_depth_to_color/image_raw \\\n"
            "--output replica/output/{} \\\n".format(time.strftime("%Y%m%d_%H%M%S"))
        ),
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('-c',"--config", type=str, default="config/rtabmap_config.yaml", help="config file path")
    parser.add_argument('-r',"--rgb_topic", type=str, default="/cam_1/color/image_raw", help="RGB image topic")
    parser.add_argument('-d',"--depth_topic", type=str, default="/cam_1/aligned_depth_to_color/image_raw", help="Depth image topic")
    parser.add_argument("--camera_info_topic", type=str, default="/cam_1/color/camera_info", help="camera info topic")
    parser.add_argument("--world_frame", type=str, default="base_link", help="world frame id")
    parser.add_argument("--camera_frame", type=str, default="cam_1_color_optical_frame", help="camera frame id")
    parser.add_argument("-o","--output", type=str, default=f"replica/output/{time.strftime('%Y%m%d_%H%M%S')}", help="output path")
    parser.add_argument("--depth_scale", type=float, default=1000.0, help="depth scale factor")
    parser.add_argument("--vis_gui",action="store_true",help="use opencv to visuliazation the whole process")
    
    args = parser.parse_args()

    config = load_config(args.config)
    torch.multiprocessing.set_start_method("spawn")

    node = OmniROSNode(args, config)
    
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down...")
    finally:
        node.terminate()
        print(f"Done, results saved to {node.output}")