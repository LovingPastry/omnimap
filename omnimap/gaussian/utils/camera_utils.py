import torch
from torch import nn
import numpy as np
import scipy.ndimage
import cv2
from gaussian.utils.graphics_utils import getProjectionMatrix2, getWorld2View2, focal2fov
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

class Camera(nn.Module):
    def __init__(
        self,
        uid,
        color,
        depth,
        gt_T,
        projection_matrix,
        fx,
        fy,
        cx,
        cy,
        fovx,
        fovy,
        image_height,
        image_width,
        device="cuda:0",
        normal=None,
        bg=[0,0,0]
    ):
        super(Camera, self).__init__()
        self.uid = uid
        self.device = device

        T = torch.eye(4, device=device)
        self.R = T[:3, :3]
        self.T = T[:3, 3]
        self.R_gt = gt_T[:3, :3]
        self.T_gt = gt_T[:3, 3]

        self.original_image = color.to(self.device)
        if depth is not None:
            self.depth = depth.to(self.device)
        if normal is not None:
            self.normal = normal.to(self.device)
        self.bg = torch.tensor(bg, dtype=torch.float32).to(self.device)
        self.grad_mask = None

        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.FoVx = fovx
        self.FoVy = fovy
        self.image_height = image_height
        self.image_width = image_width

        self.cam_rot_delta = nn.Parameter(
            torch.zeros(3, requires_grad=True, device=device)
        )
        self.cam_trans_delta = nn.Parameter(
            torch.zeros(3, requires_grad=True, device=device)
        )
        self.exposure_a = nn.Parameter(
            torch.tensor([0.0], requires_grad=True, device=device)
        )
        self.exposure_b = nn.Parameter(
            torch.tensor([0.0], requires_grad=True, device=device)
        )
        
        # for deblur the camera motion blur
        self.weight_this = nn.Parameter(
            torch.tensor([0.5], requires_grad=True, device=device)
        )
        self.weight_blur = nn.Parameter(
            torch.tensor([0.5], requires_grad=True, device=device)
        )
        self.blur_tran_x = nn.Parameter(
            torch.tensor([0.0], requires_grad=True, device=device)
        )
        self.blur_tran_y = nn.Parameter(
            torch.tensor([0.0], requires_grad=True, device=device)
        )

        self.projection_matrix = projection_matrix.to(device=device)
        
        if normal is None and depth is not None:
            self.normal = self.depth_to_normal()
        
    def set_GSRasterization(self, scaling_modifier=1.0):
        # Set up rasterization configuration
        tanfovx = math.tan(self.FoVx * 0.5)
        tanfovy = math.tan(self.FoVy * 0.5)
        raster_settings = GaussianRasterizationSettings(
            image_height=int(self.image_height),
            image_width=int(self.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=self.bg,
            scale_modifier=scaling_modifier,
            viewmatrix=self.world_view_transform,
            projmatrix=self.full_proj_transform,
            projmatrix_raw=self.projection_matrix,
            sh_degree=0,
            campos=self.camera_center
        )
        
        self.rasterizer = GaussianRasterizer(raster_settings=raster_settings)
            
    def depths_to_points(self, depth=None, world_frame=False):
        W, H = self.image_width, self.image_height
        fx = W / (2 * math.tan(self.FoVx / 2.))
        fy = H / (2 * math.tan(self.FoVy / 2.))
        intrins = torch.tensor([[fx, 0., W/2.], [0., fy, H/2.], [0., 0., 1.0]]).float().cuda()
        grid_x, grid_y = torch.meshgrid(torch.arange(W, device='cuda').float() + 0.5, torch.arange(H, device='cuda').float() + 0.5, indexing='xy')
        points = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1).reshape(-1, 3)
        if world_frame:
            c2w = (self.world_view_transform.T).inverse()
            rays_d = points @ intrins.inverse().T @ c2w[:3,:3].T
            rays_o = c2w[:3,3]
            if depth is not None:
                points = depth.reshape(-1, 1) * rays_d + rays_o
            else:
                points = self.depth.reshape(-1, 1) * rays_d + rays_o
        else:
            rays_d = points @ intrins.inverse().T
            if depth is not None:
                points = depth.reshape(-1, 1) * rays_d
            else:
                points = self.depth.reshape(-1, 1) * rays_d
        return points
    
    
    # link to camera to forbid repeated compulation
    def depth_to_normal(self, world_frame=False):
        # # bilateral smooth 
        # depth = self.depth.cpu().numpy().astype(np.float32)
        # d = 9
        # sigma_color = 555  
        # sigma_space = 555  
        # depth_smoothed = cv2.bilateralFilter(depth, d, sigma_color, sigma_space)
        # depth_smoothed = torch.tensor(depth_smoothed).to(self.depth.device)
        # points = self.depths_to_points(depth=depth_smoothed).reshape(*self.depth.shape, 3)
        points = self.depths_to_points().reshape(*self.depth.shape, 3)
        normal_map = torch.zeros_like(points)
        dx = torch.cat([points[2:, 1:-1] - points[:-2, 1:-1]], dim=0)
        dy = torch.cat([points[1:-1, 2:] - points[1:-1, :-2]], dim=1)
        normal_map[1:-1, 1:-1, :] = torch.nn.functional.normalize(torch.cross(dx, dy, dim=-1), dim=-1)
        return normal_map

    @staticmethod
    def init_from_tracking(color, depth, pose, idx, projection_matrix, K, tstamp=None, normal=None, bg=[0,0,0]):
        cam = Camera(
            idx,
            color,
            depth,
            pose,
            projection_matrix,
            K[0],
            K[1],
            K[2],
            K[3],
            focal2fov(K[0], K[-2]),
            focal2fov(K[1], K[-1]),
            K[-1],
            K[-2],
            normal=normal,
            bg=bg)
        cam.R = pose[:3, :3]
        cam.T = pose[:3, 3]
        cam.tstamp = tstamp
        cam.set_GSRasterization()
        return cam
    
    @staticmethod
    def init_from_dataset(dataset, idx, projection_matrix):
        gt_color, gt_depth, gt_pose = dataset[idx]
        return Camera(
            idx,
            gt_color,
            gt_depth,
            None,
            gt_pose,
            projection_matrix,
            dataset.fx,
            dataset.fy,
            dataset.cx,
            dataset.cy,
            dataset.fovx,
            dataset.fovy,
            dataset.height,
            dataset.width,
            # device=dataset.device,
        )

    @staticmethod
    def init_from_gui(uid, T, FoVx, FoVy, fx, fy, cx, cy, H, W, device="cuda:0"):
        device = torch.device(device) if not isinstance(device, torch.device) else device
        projection_matrix = getProjectionMatrix2(
            znear=0.01, zfar=100.0, fx=fx, fy=fy, cx=cx, cy=cy, W=W, H=H
        ).transpose(0, 1)
        return Camera(
            uid, None, None, T, projection_matrix, fx, fy, cx, cy, FoVx, FoVy, H, W,
            device=device
        )

    @property
    def world_view_transform(self):
        return getWorld2View2(self.R, self.T).transpose(0, 1)

    @property
    def full_proj_transform(self):
        return (
            self.world_view_transform.unsqueeze(0).bmm(
                self.projection_matrix.unsqueeze(0)
            )
        ).squeeze(0)

    @property
    def camera_center(self):
        return self.world_view_transform.inverse()[3, :3]

    def update_RT(self, R, t):
        self.R = R.to(device=self.device)
        self.T = t.to(device=self.device)

    def clean(self):
        self.original_image = None
        self.depth = None
        self.grad_mask = None

        self.cam_rot_delta = None
        self.cam_trans_delta = None

    def clone(self):
        """安全的深拷贝方法"""
        # 克隆基础数据
        cloned = Camera(
            uid=self.uid,
            color=self.original_image.clone().detach(),
            depth=self.depth.clone().detach() if self.depth is not None else None,
            gt_T=torch.cat([
                torch.cat([self.R_gt, self.T_gt.unsqueeze(1)], dim=1),
                torch.tensor([[0, 0, 0, 1]], device=self.device)
            ], dim=0),
            projection_matrix=self.projection_matrix.clone(),
            fx=self.fx, fy=self.fy, cx=self.cx, cy=self.cy,
            fovx=self.FoVx, fovy=self.FoVy,
            image_height=self.image_height,
            image_width=self.image_width,
            device=self.device,
            normal=self.normal.clone().detach() if self.normal is not None else None,
            bg=self.bg.cpu().tolist()
        )
        
        # 克隆当前姿态
        cloned.R = self.R.clone()
        cloned.T = self.T.clone()
        
        # 克隆参数（不带梯度）
        with torch.no_grad():
            cloned.cam_rot_delta.copy_(self.cam_rot_delta)
            cloned.cam_trans_delta.copy_(self.cam_trans_delta)
            cloned.exposure_a.copy_(self.exposure_a)
            cloned.exposure_b.copy_(self.exposure_b)
            cloned.weight_this.copy_(self.weight_this)
            cloned.weight_blur.copy_(self.weight_blur)
            cloned.blur_tran_x.copy_(self.blur_tran_x)
            cloned.blur_tran_y.copy_(self.blur_tran_y)
        
        # 复制其他属性
        if hasattr(self, 'tstamp'):
            cloned.tstamp = self.tstamp
        if self.grad_mask is not None:
            cloned.grad_mask = self.grad_mask.clone()
        
        cloned.set_GSRasterization()
        return cloned
    
    def __deepcopy__(self, memo):
        """自定义深拷贝行为"""
        return self.clone()

class HemisphereCamera(Camera):
    """
    半球约束相机：位置由(theta, phi)与半径确定，始终朝向球心。
    角度单位：弧度
    theta: 0..2π
    phi: 0..π/2 (上半球)
    """
    def __init__(
        self,
        uid,
        color,
        depth,
        gt_T,
        projection_matrix,
        fx,
        fy,
        cx,
        cy,
        fovx,
        fovy,
        image_height,
        image_width,
        center,   # 球心坐标 (3,)
        radius,   # 半径
        theta,  # 方位角
        phi,# 仰角
        device="cuda:0",
        normal=None,
        bg=[0, 0, 0]
    ):
        super().__init__(
            uid, color, depth, gt_T, projection_matrix, fx, fy, cx, cy,
            fovx, fovy, image_height, image_width, device=device, normal=normal, bg=bg
        )
        self.center = torch.tensor(center, dtype=torch.float32, device=device)
        self.radius = float(radius)

        self.theta = nn.Parameter(torch.tensor([theta], dtype=torch.float32, device=device))
        self.phi = nn.Parameter(torch.tensor([phi], dtype=torch.float32, device=device))

        self.update_pose_from_spherical()

    def update_pose_from_spherical(self):
        # 约束仰角到 [0, pi/2]
        phi = torch.clamp(self.phi, 0.0, math.pi / 2.0)
        theta = self.theta

        # 球坐标 -> 笛卡尔
        x = self.radius * torch.cos(phi) * torch.cos(theta)
        y = self.radius * torch.sin(phi)
        z = self.radius * torch.cos(phi) * torch.sin(theta)
        cam_pos = self.center + torch.stack([x, y, z]).reshape(3)

        # 计算朝向球心的旋转矩阵
        # 相机 -Z 轴指向球心（look-at）
        forward = torch.nn.functional.normalize(self.center - cam_pos, dim=0)
        world_up = torch.tensor([0.0, 1.0, 0.0], device=self.device)
        right = torch.nn.functional.normalize(torch.cross(world_up, forward), dim=0)
        up = torch.cross(forward, right)

        # 构造世界到相机的 R/T
        # 这里采用列向量为相机坐标轴的约定
        R = torch.stack([right, up, forward], dim=1)  # 3x3
        T = cam_pos

        self.update_RT(R, T)

    def set_angles(self, theta=None, phi=None):
        with torch.no_grad():
            if theta is not None:
                self.theta.copy_(torch.tensor([theta], device=self.device))
            if phi is not None:
                self.phi.copy_(torch.tensor([phi], device=self.device))
        self.update_pose_from_spherical()
    
    def get_angles(self):
        return self.theta.item(), self.phi.item()

    def to_camera(self) -> Camera:
        """
        将 HemisphereCamera 转换回普通 Camera
        
        Returns:
            camera: 普通 Camera 对象
        """
        camera = Camera(
            uid=self.uid,
            color=self.original_image,
            depth=self.depth if hasattr(self, 'depth') else None,
            gt_T=torch.eye(4, device=self.device),
            projection_matrix=self.projection_matrix,
            fx=self.fx,
            fy=self.fy,
            cx=self.cx,
            cy=self.cy,
            fovx=self.FoVx,
            fovy=self.FoVy,
            image_height=self.image_height,
            image_width=self.image_width,
            device=self.device,
            normal=self.normal if hasattr(self, 'normal') else None,
            bg=self.bg.cpu().tolist()
        )
        
        # 设置当前的 R 和 T
        camera.R = self.R.clone()
        camera.T = self.T.clone()
        
        # 复制其他属性
        if hasattr(self, 'tstamp'):
            camera.tstamp = self.tstamp
        
        if hasattr(self, 'grad_mask'):
            camera.grad_mask = self.grad_mask
        
        # 复制优化参数
        with torch.no_grad():
            camera.cam_rot_delta.copy_(self.cam_rot_delta)
            camera.cam_trans_delta.copy_(self.cam_trans_delta)
            camera.exposure_a.copy_(self.exposure_a)
            camera.exposure_b.copy_(self.exposure_b)
        
        camera.set_GSRasterization()
        
        return camera
    
    @classmethod
    def from_camera(
        cls,
        camera: Camera,
        center: torch.Tensor,
        radius: float = None,
        auto_radius: bool = True
    ) -> 'HemisphereCamera':
        """
        从 Camera 对象构造 HemisphereCamera, 根据相机位置计算 theta 和 phi。
        Example:
            >>> cam = Camera.init_from_tracking(...)
            >>> hemi_cam = HemisphereCamera.from_camera(cam, center=[0, 0, 0])
        """
        device = camera.device
        
        # 确保 center 是 tensor
        if not isinstance(center, torch.Tensor):
            center = torch.tensor(center, dtype=torch.float32, device=device)
        else:
            center = center.to(device=device)
        
        # 获取相机位置
        cam_pos = camera.camera_center
        
        # 计算半径
        if radius is None or auto_radius:
            radius = torch.norm(cam_pos - center).item()
            if radius < 1e-6:
                raise ValueError("Camera is too close to the sphere center!")
        
        # 计算球坐标
        rel_pos = cam_pos - center
        r = torch.norm(rel_pos)
        if r < 1e-6:
            raise ValueError("Camera position coincides with sphere center!")
        
        rel_pos_normalized = rel_pos / r
        phi = torch.asin(torch.clamp(rel_pos_normalized[1], -1.0, 1.0))
        theta = torch.atan2(rel_pos_normalized[2], rel_pos_normalized[0])
        
        # 调用标准构造函数
        instance = cls(
            uid=camera.uid,
            color=camera.original_image,
            depth=camera.depth if hasattr(camera, 'depth') else None,
            gt_T=torch.eye(4, device=device),
            projection_matrix=camera.projection_matrix,
            fx=camera.fx,
            fy=camera.fy,
            cx=camera.cx,
            cy=camera.cy,
            fovx=camera.FoVx,
            fovy=camera.FoVy,
            image_height=camera.image_height,
            image_width=camera.image_width,
            center=center.cpu().numpy(),
            radius=radius,
            theta=theta.item(),
            phi=phi.item(),
            device=device,
            normal=camera.normal if hasattr(camera, 'normal') else None,
            bg=camera.bg.cpu().tolist() if hasattr(camera, 'bg') else [0, 0, 0]
        )
        
        # 复制额外属性
        if hasattr(camera, 'tstamp'):
            instance.tstamp = camera.tstamp
        if hasattr(camera, 'grad_mask'):
            instance.grad_mask = camera.grad_mask
        
        # 复制优化参数
        with torch.no_grad():
            if hasattr(camera, 'cam_rot_delta'):
                instance.cam_rot_delta.copy_(camera.cam_rot_delta)
            if hasattr(camera, 'cam_trans_delta'):
                instance.cam_trans_delta.copy_(camera.cam_trans_delta)
            if hasattr(camera, 'exposure_a'):
                instance.exposure_a.copy_(camera.exposure_a)
            if hasattr(camera, 'exposure_b'):
                instance.exposure_b.copy_(camera.exposure_b)
        
        instance.set_GSRasterization()
        return instance
    

    def clone(self):
        """安全的深拷贝方法"""
        cloned = HemisphereCamera(
            uid=self.uid,
            color=self.original_image.clone().detach(),
            depth=self.depth.clone().detach() if self.depth is not None else None,
            gt_T=torch.eye(4, device=self.device),
            projection_matrix=self.projection_matrix.clone(),
            fx=self.fx, fy=self.fy, cx=self.cx, cy=self.cy,
            fovx=self.FoVx, fovy=self.FoVy,
            image_height=self.image_height,
            image_width=self.image_width,
            center=self.center.cpu().numpy(),
            radius=self.radius,
            theta=self.theta.item(),
            phi=self.phi.item(),
            device=self.device,
            normal=self.normal.clone().detach() if self.normal is not None else None,
            bg=self.bg.cpu().tolist()
        )
        
        # 复制父类参数
        with torch.no_grad():
            cloned.cam_rot_delta.copy_(self.cam_rot_delta)
            cloned.cam_trans_delta.copy_(self.cam_trans_delta)
            cloned.exposure_a.copy_(self.exposure_a)
            cloned.exposure_b.copy_(self.exposure_b)
        
        if hasattr(self, 'tstamp'):
            cloned.tstamp = self.tstamp
        if self.grad_mask is not None:
            cloned.grad_mask = self.grad_mask.clone()
        
        cloned.set_GSRasterization()
        return cloned
    
    def __deepcopy__(self, memo):
        """自定义深拷贝行为"""
        return self.clone()