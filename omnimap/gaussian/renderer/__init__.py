#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from gaussian.scene.gaussian_model import GaussianModel
from gaussian.utils.camera_utils import Camera


def render(viewpoint_camera:Camera, pc : GaussianModel, bg_color : torch.Tensor, scaling_modifier = 1.0):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """ 
    
    # # Set up rasterization configuration
    # tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    # tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    
    # raster_settings = GaussianRasterizationSettings(
    #     image_height=int(viewpoint_camera.image_height),
    #     image_width=int(viewpoint_camera.image_width),
    #     tanfovx=tanfovx,
    #     tanfovy=tanfovy,
    #     bg=bg_color,
    #     scale_modifier=scaling_modifier,
    #     viewmatrix=viewpoint_camera.world_view_transform,
    #     projmatrix=viewpoint_camera.full_proj_transform,
    #     projmatrix_raw=viewpoint_camera.projection_matrix,
    #     sh_degree=pc.active_sh_degree,
    #     campos=viewpoint_camera.camera_center
    # )

    # rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass
    
    rasterizer = viewpoint_camera.rasterizer
    means3D = pc.get_xyz
    means2D = screenspace_points

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    cov3D_precomp = None
    scales = pc.get_scaling
    opacity = pc.get_opacity
    rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = pc.get_features
    colors_precomp = None
    
    # print("init time", time.time()-start_time) 
    # start_time = time.time()
    
    rendered_image, radii, rendered_expected_depth, n_touched = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp,
        theta = viewpoint_camera.cam_rot_delta,
        rho = viewpoint_camera.cam_trans_delta,
    )
    normal_image = _render_normals_helper(viewpoint_camera, pc, rasterizer)
    # print("acc render time", time.time()-start_time) 

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "depth": rendered_expected_depth,
            "viewspace_points": means2D,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "n_touched": n_touched,
            "normal": normal_image}




def render_instance(viewpoint_camera, pc : GaussianModel, bg_color = torch.Tensor([0.2,0.2,0.2]).cuda(), scaling_modifier = 1.0):
    """
    Render the instance scene.
    Background tensor (bg_color) must be on GPU!
    """
    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        projmatrix_raw=viewpoint_camera.projection_matrix,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    cov3D_precomp = None
    scales = pc.get_scaling
    opacity = pc.get_opacity
    rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = pc.get_instance_features
    colors_precomp = None

    rendered_ins_image, _, _, _ = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp,
        theta = viewpoint_camera.cam_rot_delta,
        rho = viewpoint_camera.cam_trans_delta,
    )

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_ins_image}



def _render_normals_helper(viewpoint_camera, pc, rasterizer):
    """辅助函数：渲染法向图"""
    cov = pc.get_covariance_matrices()
    eigenvalues, eigenvectors = torch.linalg.eigh(cov)
    normals_world = eigenvectors[:, :, 0]
    
    # 转换到相机坐标系
    R_cam = viewpoint_camera.world_view_transform[:3, :3]
    normals_cam = (R_cam @ normals_world.T).T
    normals_cam = normals_cam / (torch.norm(normals_cam, dim=1, keepdim=True) + 1e-7)
    
    # RGB编码
    normals_rgb = (normals_cam + 1) / 2
    
    # 创建临时颜色特征
    colors_precomp = normals_rgb
    
    # 重新渲染（复用光栅化器设置）
    normal_image, _, _, _ = rasterizer(
        means3D=pc.get_xyz,
        means2D=torch.zeros_like(pc.get_xyz[:, :2]),
        shs=None,
        colors_precomp=colors_precomp,
        opacities=pc.get_opacity,
        scales=pc.get_scaling,
        rotations=pc.get_rotation,
        cov3D_precomp=None,
        theta=viewpoint_camera.cam_rot_delta,
        rho=viewpoint_camera.cam_trans_delta,
    )
    
    return normal_image