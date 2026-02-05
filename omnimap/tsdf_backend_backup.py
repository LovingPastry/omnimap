
import torch
import os,sys
import warnings
warnings.filterwarnings('ignore')
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch.nn.functional as F
import matplotlib.pyplot as plt

from scipy.spatial import KDTree
import distinctipy

import open3d as o3d
import open3d.core as o3c
from util.utils import Log
import numpy as np
import torch

from .visual_module import VisualModule

class TSDFBackEnd():
    def __init__(self, config, save_dir, vis_gui):
        super().__init__()
        self.device = torch.device("cuda")
        self.o3c_device = o3c.Device("CUDA:0")
        self.vis_gui = vis_gui
        self.save_dir = save_dir
        self.config = config
        self.voxel_size = self.config["tsdf"]["voxel_size"]
        self.block_resolution = self.config["tsdf"]["block_resolution"]
        self.block_count = self.config["tsdf"]["block_count"]
        self.unregistered_threshold = self.config["tsdf"]["unregistered_threshold"]
        self.world = o3d.t.geometry.VoxelBlockGrid(
            ('tsdf', 'weight', 'color'),
            (o3c.float32, o3c.float32, o3c.float32),
            ((1), (1), (3)),
            self.voxel_size,
            self.block_resolution,
            self.block_count, device=self.o3c_device
        )
        self.instance_id_vol = torch.zeros(self.block_count, self.block_resolution, self.block_resolution, self.block_resolution, 3).long().to(self.device)
        self.instance_pro_vol = torch.zeros(self.block_count, self.block_resolution, self.block_resolution, self.block_resolution, 4).to(torch.uint8).to(self.device)
        self.instance_count_vol = torch.zeros(self.block_count, self.block_resolution, self.block_resolution, self.block_resolution).to(torch.uint8).to(self.device)
        self.instance_feature = torch.zeros((1,384), device=self.device)
        self.instance_fea_count = torch.ones((1, 1), dtype=torch.int64, device=self.device)
        self.instance_fea_weight = torch.ones((1, 1), dtype=torch.float, device=self.device)
        self.unregistered_mask = self.instance_count_vol > 100.0 # Full False for initial
        self.new_voxel = self.instance_count_vol < 100.0 # Full True for initial
        self.last_num = 0
        
        self.instance_skip = self.config["instance"]["instance_skip"]
        self.pro_thre = self.config["instance"]["pro_thre"]
        self.ins_min_count = self.config["instance"]["ins_min_count"]
        self.vox_min_count = self.config["instance"]["vox_min_count"]
        self.ins_min_voxel = self.config["instance"]["ins_min_voxel"]
        self.fill_dis = self.config["instance"]["fill_dis"]
        
        self.intrinsic = None
        self.x_coords = None
        self.height = None
        self.all_pc = o3d.geometry.PointCloud()
        
        self.load_models()
        
        instance_file = "pretrained_models/instance_colors.pt"
        if os.path.exists(instance_file):
            self.instance_colors = torch.load(instance_file).cuda().to(torch.int64)
        else:
            instance_colors = distinctipy.get_colors(1000, pastel_factor=1.0)
            instance_colors[0] = [0.2, 0.2, 0.2] # for background
            self.instance_colors = torch.tensor(instance_colors, dtype=torch.float32).cuda()*255
            self.instance_colors = self.instance_colors.to(torch.int64)
            torch.save(self.instance_colors, instance_file)
            
        if self.vis_gui:
            self.o3d_window = None
        
    def update_vis(self):
        if self.o3d_window is None:
            self.o3d_window = o3d.visualization.VisualizerWithKeyCallback()
            self.o3d_window.create_window(window_name="Instance Viewer", width=860, height=540)
            self.instance_vis = None
        if self.instance_vis is not None:
            self.o3d_window.remove_geometry(self.instance_vis)
        self.instance_vis = self.vis_instance()
        self.o3d_window.add_geometry(self.instance_vis)
        self.o3d_window.poll_events()
        self.o3d_window.update_renderer()
        
    def vis_instance(self):        
        buf_indices = self.world.hashmap().active_buf_indices()
        voxel_coords, voxel_indices = self.world.voxel_coordinates_and_flattened_indices(buf_indices)
        o3c.cuda.synchronize()
        voxel_coords, _ = self.world.voxel_coordinates_and_flattened_indices(buf_indices)
        buf_indices = torch.utils.dlpack.from_dlpack(buf_indices.to_dlpack()) # (M,)
        voxel_coords = torch.utils.dlpack.from_dlpack(voxel_coords.to_dlpack()) # (Mx8x8x8,3)
        voxel_coords_ok = voxel_coords.view(-1, self.block_resolution, self.block_resolution, self.block_resolution, 3)  # (M,8,8,8,3)
        buf_instance_count_vol = self.instance_count_vol[buf_indices]
        count_mask = buf_instance_count_vol>1
        points = voxel_coords_ok[count_mask]
        buf_instance_pro_vol = self.instance_pro_vol[buf_indices]
        max_pro_indices = torch.argmax(buf_instance_pro_vol[...,:3], dim=-1, keepdim=True)
        buf_instance_id_vol = self.instance_id_vol[buf_indices]
        max_pro_instance_id = torch.gather(buf_instance_id_vol, dim=-1, index=max_pro_indices).squeeze(-1)[count_mask].to(torch.long)
        # remove the small label
        unique_labels, counts = torch.unique(max_pro_instance_id, return_counts=True)
        labels_to_remove_min_voxel = unique_labels[counts < self.ins_min_voxel]
        unique_labels_count = self.instance_fea_count[unique_labels][:,0]
        labels_to_remove_min_count = unique_labels[unique_labels_count < self.ins_min_count]
        labels_to_remove = torch.unique(torch.cat((labels_to_remove_min_voxel, labels_to_remove_min_count)))
        mask = torch.isin(max_pro_instance_id, labels_to_remove)
        points = points[~mask]
        max_pro_instance_id = max_pro_instance_id[~mask]
        ins_colors = torch.index_select(self.instance_colors, 0, max_pro_instance_id)/255.0
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(points.cpu().numpy())
        pc.colors = o3d.utility.Vector3dVector(ins_colors.cpu().numpy())
        return pc
        
            
        
    def load_models(self): 
        # 初始化视觉模块，封装了YOLO-World、TAP和SBERT模型
        self.visual_module = VisualModule(self.config)
        
    def get_mesh(self, legacy=True):
        mesh = self.world.extract_triangle_mesh()
        return mesh.to_legacy() if legacy else mesh
    
    
    @staticmethod
    def depth_to_point_cloud(depth, extrinsic, intrinsic, image_width, image_height, depth_scale):
        """
        Args:
            depth (np.array): depth image
            extrinsic (o3c.Tensor): shape of (4, 4)
            intrinsic (o3c.Tensor): shape of (3, 3). Use self.custom_intrinsic(image_width, image_height)
            image_width (int): image width
            image_height (int): image height
            depth_max (float): depth max
            depth_scale (float): depth scale
        Returns:
            coords (torch.Tensor): shape of (N, 3)
            mask (torch.Tensor): shape of (height, width)
        """
        depth = torch.from_numpy(depth.astype(np.int32)) / depth_scale
        depth = F.interpolate(
            depth.unsqueeze(0).unsqueeze(0).float(),
            (image_height, image_width)
        ).view(image_height, image_width).cuda()
        extrinsic = torch.utils.dlpack.from_dlpack(extrinsic.to_dlpack()).cuda().float()
        intrinsic = torch.utils.dlpack.from_dlpack(intrinsic.to_dlpack()).cuda().float()
        fx, fy, cx, cy = intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]
        v, u = torch.meshgrid(torch.arange(image_height).cuda(), torch.arange(image_width).cuda(), indexing="ij")
        uvd = torch.stack([u, v, torch.ones_like(depth)], dim=0).float() # (3,height,width)
        # NOTE: don't use torch.inverse(intrinsic) as it is slow
        uvd[0] = (uvd[0] - cx) / fx
        uvd[1] = (uvd[1] - cy) / fy
        xyz = uvd.view(3, -1) * depth.view(1, -1) # (3, height*width)
        # NOTE: convert to world frame
        R = extrinsic[:3, :3].T
        coords =  (R @ xyz - R @ extrinsic[:3, 3:]).view(3, image_height, image_width).permute(1,2,0)
        mask = depth>0.01
        return coords.permute(1, 0, 2).reshape(-1,3), mask.permute(1, 0).reshape(-1)
    
    
    
    def find_buf_indices_from_coord(self, buf_indices, voxel_coords, coordinates):
        """
        Finds the index of the cube that incorporates each coordinate in a batched manner.

        Args:
            buf_indices (torch.Tensor): N tensor of buf indices for the given voxel coords
            voxel_coords (torch.Tensor): Nx8^3x3 tensor where N is the number of cubes.
            coordinates (torch.Tensor): Mx3 tensor where M is the number of coordinates. (usually M >> N)

        Returns:
            tensor: M tensor that contains the index of the cube that incorporates each coordinate.
        """
        # NOTE: find min and max of x, y, z for each cube
        # min_vals = torch.min(voxel_coords, dim=1).values  # Shape: M
        # max_vals = torch.max(voxel_coords, dim=1).values  # Shape: M
        # NOTE: account for border coordinates
        min_vals = voxel_coords[:, 0, :]  - self.voxel_size/2 # Shape: M
        max_vals = voxel_coords[:, -1, :] + self.voxel_size/2 # Shape: M
        # NOTE: check if each coordinate is inside each cube
        is_inside = (min_vals[:, None] <= coordinates[None]) & (coordinates[None] < max_vals[:, None]) # Shape: MxNx3
        
        # NOTE: all coordinates must be inside the cube along all 3 dimensions (x, y, z)
        # is_inside_all_dims = torch.all(is_inside, dim=2).long()  # Shape: MxN
        is_inside_all_dims = torch.all(is_inside, dim=2).to(torch.uint8) # Shape: MxN

        # find the block index
        max_index = torch.argmax(is_inside_all_dims, dim=0)
        cube_indices = buf_indices[max_index]  # Shape: N
        # the local voxel indices（x, y, z）
        local_voxel_indices = torch.floor((coordinates - min_vals[max_index]) / self.voxel_size).long()  # Shape: Nx3
        local_voxel_indices[:, [0, 2]] = local_voxel_indices[:, [2, 0]]
        # NOTE: find valid mask where a cube was found for a coordinate
        valid_mask = torch.any(is_inside_all_dims, dim=0)  # Shape: N
        return cube_indices, local_voxel_indices, max_index, valid_mask
    
    


            
    def integrate(self, color_im, depth_im, cam_intr, cam_pose, tstamp, obs_weight=1.0):
        self.depth_im = depth_im.cuda()
        color = o3d.t.geometry.Image(np.ascontiguousarray(color_im.cpu().numpy())).to(o3c.uint8).to(self.o3c_device)
        depth_im = depth_im * 1000.0
        depth = o3d.t.geometry.Image(np.ascontiguousarray(depth_im.cpu().numpy())).to(o3c.uint16).to(self.o3c_device)
        self.cam_pose = torch.inverse(cam_pose).to(self.device)
        cam_pose = np.linalg.inv(cam_pose.cpu().numpy())
        extrinsic = o3c.Tensor.from_numpy(cam_pose)
        # Get active frustum block coordinates from input
        if self.intrinsic is None:
            self.intrinsic_raw = torch.tensor([cam_intr[0], 0.0, cam_intr[2], 0.0, cam_intr[1], cam_intr[3], 0.0, 0.0, 1.0]).to(self.device).reshape(3,3)
            cam_intr = cam_intr.cpu().numpy()
            intrinsic_np =  np.array([cam_intr[0], 0.0, cam_intr[2], 0.0, cam_intr[1], cam_intr[3], 0.0, 0.0, 1.0]).reshape(3,3)
            self.intrinsic = o3c.Tensor.from_numpy(intrinsic_np)
        # max_depth have been valued
        frustum_block_coords = self.world.compute_unique_block_coordinates(depth, self.intrinsic, extrinsic, 1000.0, 20.0)
        self.world.integrate(frustum_block_coords, depth, color, self.intrinsic, extrinsic, 1000.0, 20.0)
        
        
        self.adjust_embed_capacity()
        # all block indices in view frustum
        cur_buf_indices, _ = self.world.hashmap().find(frustum_block_coords) # (M,)
        o3c.cuda.synchronize()
        # all voxel_coords and voxel_indices that belong to cur_buf_indices block
        voxel_coords, _ = self.world.voxel_coordinates_and_flattened_indices(cur_buf_indices)
        # voxel_indices = self.world.voxel_indices(cur_buf_indices)
        
        cur_buf_indices = torch.utils.dlpack.from_dlpack(cur_buf_indices.to_dlpack()) # (M,)
        voxel_coords = torch.utils.dlpack.from_dlpack(voxel_coords.to_dlpack()) # (Mx8x8x8,3)
        self.voxel_coords_ok = voxel_coords.view(-1, self.block_resolution, self.block_resolution, self.block_resolution, 3)  # (M,8,8,8,3)
        # voxel_indices = torch.utils.dlpack.from_dlpack(voxel_indices.to_dlpack())
        # voxel_indices = voxel_indices.transpose(0, 1) # (Mx8x8x8,4)
        
        if self.height is None:
            self.height, self.width = depth_im.shape[:2]
        # (N, 3)       (N,)
        self.obs_coords, depth_valid = self.depth_to_point_cloud(depth_im.cpu().numpy(), extrinsic, self.intrinsic, self.width, self.height, 1000.0)
        

        # (N,)            (N,3)                        (N,)                             (N,)
        cube_indices, local_voxel_indices, cube_indices_indices, block_valid = self.find_buf_indices_from_coord(
            cur_buf_indices, # (M,)
            voxel_coords.view(-1, self.block_resolution**3, 3), # (M, 8^3, 3)
            self.obs_coords # (N, 3)
        )
        # full indices (N, 5) (1 for global block, 1 for local block, 3 for voxel)
        self.combined_indices = torch.cat((cube_indices.unsqueeze(1), cube_indices_indices.unsqueeze(1), local_voxel_indices), dim=1)
        self.combined_indices[:, 2] = torch.clamp(self.combined_indices[:, 2], min=0, max=self.block_resolution-1)
        self.combined_indices[:, 3] = torch.clamp(self.combined_indices[:, 3], min=0, max=self.block_resolution-1)
        self.combined_indices[:, 4] = torch.clamp(self.combined_indices[:, 4], min=0, max=self.block_resolution-1)
        self.all_vaild = depth_valid & block_valid
        
        self.tstamp = tstamp
        if tstamp%self.instance_skip != 0:
            return
        
        img = color_im.cpu().numpy()[:,:,::-1]
        
        # 使用视觉模块进行目标检测和分割
        final_masks, caption_fts, last_mask_image = self.visual_module.detect(img, vis_gui=self.vis_gui)
        
        # 如果没有检测到物体，返回
        if len(final_masks) == 0:
            return
        
        # 更新可视化图像
        if self.vis_gui and last_mask_image is not None:
            self.last_mask_image = last_mask_image
        
        '''[4] get 3D voxels'''
        
        # current scene state
        count_mask = self.instance_count_vol>self.vox_min_count
        max_pro_indices = torch.argmax(self.instance_pro_vol[...,:3], dim=-1, keepdim=True) 
        max_pro_instance_id = torch.gather(self.instance_id_vol, dim=-1, index=max_pro_indices).squeeze(-1)[count_mask].to(torch.long)
        
        # Iterate through each instance mask
        for mask_id in range(final_masks.shape[0]):
            # print("*"*100)
            # associated voxels
            
            this_instance_image_mask = final_masks[mask_id].T.flatten() & self.all_vaild
            
            instance_indices = self.combined_indices[this_instance_image_mask]
            
            instance_indices = torch.unique(instance_indices, dim=0)
            
            # denoise
            coords = self.voxel_coords_ok[instance_indices[:,1], instance_indices[:,2], instance_indices[:,3], instance_indices[:,4]]
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(coords.cpu().numpy())
            cl, ind = pcd.remove_statistical_outlier(nb_neighbors=10, std_ratio=2.0)
            instance_indices = instance_indices[ind]
            
            occupied_voxels_num = len(instance_indices)
            instance_vol_id = self.instance_id_vol[instance_indices[:,0], instance_indices[:,2],  instance_indices[:,3], instance_indices[:,4]]
            exist_instance_ids = torch.unique(instance_vol_id[instance_vol_id != 0])
            # no informations, new instance for init
            if len(exist_instance_ids) == 0:
                self.instance_id_vol[instance_indices[:,0], instance_indices[:,2],  instance_indices[:,3], instance_indices[:,4], 0] = len(self.instance_feature)
                self.instance_pro_vol[instance_indices[:,0], instance_indices[:,2],  instance_indices[:,3], instance_indices[:,4], 0] += 1
                self.instance_feature = torch.cat((self.instance_feature, caption_fts[mask_id].unsqueeze(0)), dim = 0)
                self.instance_fea_count = torch.cat((self.instance_fea_count, torch.tensor([1]).unsqueeze(0).cuda()), dim = 0)
                self.instance_fea_weight = torch.cat((self.instance_fea_weight, torch.tensor([1]).unsqueeze(0).cuda()), dim = 0)
                self.instance_count_vol[instance_indices[:,0], instance_indices[:,2],  instance_indices[:,3], instance_indices[:,4]] += 1
            else:
                # look for each possible instance id for current mask
                geometry_sim = []
                for id in exist_instance_ids:
                    indices = torch.nonzero(instance_vol_id == id)
                    # joint probabilites for id (for simply, use add)
                    this_id_pro_count = self.instance_pro_vol[instance_indices[:,0], instance_indices[:,2],  instance_indices[:,3], \
                                                                                        instance_indices[:,4]][[indices[:, 0], indices[:, 1]]]
                    this_id_all_count = self.instance_count_vol[instance_indices[:,0], instance_indices[:,2],  instance_indices[:,3], \
                                                                                        instance_indices[:,4]][indices[:, 0]]
                    this_id_pro = this_id_pro_count.float() / this_id_all_count.float()
                    geometry_probability = this_id_pro.sum() / occupied_voxels_num
                    geometry_sim.append(geometry_probability)
                    
                geometry_sim = torch.stack(geometry_sim)
                # feature similarity
                this_instance_fea = caption_fts[mask_id]
                exist_instances_fea = self.instance_feature[exist_instance_ids]
                feature_sim = F.cosine_similarity(exist_instances_fea, this_instance_fea.unsqueeze(0), dim=1)
                # overall similartity
                overall_sim = geometry_sim * 0.8 + feature_sim * 0.2
                associated_id = None
                max_pro, max_index = torch.max(overall_sim, dim=0)
                if max_pro>self.pro_thre:
                    associated_id = exist_instance_ids[max_index]
                    
                    
                if associated_id is not None:
                    # visual ratio
                    associated_instance_volume = (max_pro_instance_id==associated_id).sum().float()
                    vis_ratio = torch.clip(occupied_voxels_num/associated_instance_volume, min=0., max=1.)*max_pro
                    # update the feature
                    self.instance_feature[associated_id] = (self.instance_feature[associated_id] * self.instance_fea_weight[associated_id] + \
                                                                                this_instance_fea * vis_ratio ) / (self.instance_fea_weight[associated_id] + vis_ratio)
                    
                    self.instance_fea_count[associated_id] += 1
                    self.instance_fea_weight[associated_id] += vis_ratio
                    
                    associated_vol_indices = torch.nonzero(instance_vol_id == associated_id)
                    self.instance_pro_vol[instance_indices[:,0][associated_vol_indices[:, 0]], instance_indices[:,2][associated_vol_indices[:, 0]],\
                                                    instance_indices[:,3][associated_vol_indices[:, 0]], instance_indices[:,4][associated_vol_indices[:, 0]], associated_vol_indices[:, 1]] += 1
                    # a new probable instance_id is added for the voxles with an empty space
                    if len(associated_vol_indices) < len(instance_vol_id):
                        mask = torch.all(instance_vol_id != associated_id, dim=1)
                        instance_ids = self.instance_id_vol[instance_indices[:,0], instance_indices[:,2], instance_indices[:,3], instance_indices[:,4]][mask]
                        is_zero = (instance_ids == 0).to(torch.int64)
                        the_first_zero_indices = torch.argmax(is_zero, dim=1)
                        have_full = is_zero.sum(dim=1) == 0
                        the_first_zero_indices[have_full] = self.instance_id_vol.size(-1)  # if the instance id list is full, assigned to others
                        # add the instance id to instance_id_vol
                        no_full_mask = mask.clone()
                        true_indices = no_full_mask.nonzero() 
                        mask_M_true_indices = have_full.nonzero()
                        true_indices_in_mask_M = true_indices[mask_M_true_indices[:, 0]]
                        no_full_mask[true_indices_in_mask_M[:, 0]] = False
                        self.instance_id_vol[instance_indices[:,0][no_full_mask], instance_indices[:,2][no_full_mask], \
                                                    instance_indices[:,3][no_full_mask], instance_indices[:,4][no_full_mask], the_first_zero_indices[~have_full]] = associated_id
                        # update the probablity
                        self.instance_pro_vol[instance_indices[:,0][mask], instance_indices[:,2][mask], instance_indices[:,3][mask], instance_indices[:,4][mask] \
                                                    , the_first_zero_indices] += 1
                else: # this is a new instance
                    new_instance_id = len(self.instance_feature)
                    self.instance_feature = torch.cat((self.instance_feature, caption_fts[mask_id].unsqueeze(0)), dim = 0)
                    self.instance_fea_count = torch.cat((self.instance_fea_count, torch.tensor([1]).unsqueeze(0).cuda()), dim = 0)
                    self.instance_fea_weight = torch.cat((self.instance_fea_weight, torch.tensor([1]).unsqueeze(0).cuda()), dim = 0)
                    instance_ids = self.instance_id_vol[instance_indices[:,0], instance_indices[:,2], instance_indices[:,3], instance_indices[:,4]]
                    is_zero = (instance_ids == 0).to(torch.int64)
                    the_first_zero_indices = torch.argmax(is_zero, dim=1)
                    have_full = is_zero.sum(dim=1) == 0
                    the_first_zero_indices[is_zero.sum(dim=1) == 0] = 3  # if the instance id list is full, assigned to others
                    self.instance_id_vol[instance_indices[:,0][~have_full], instance_indices[:,2][~have_full], instance_indices[:,3][~have_full], instance_indices[:,4][~have_full] \
                                                , the_first_zero_indices[~have_full]] = new_instance_id
                    self.instance_pro_vol[instance_indices[:,0], instance_indices[:,2], instance_indices[:,3], instance_indices[:,4], the_first_zero_indices] += 1
                self.instance_count_vol[instance_indices[:,0], instance_indices[:,2], instance_indices[:,3], instance_indices[:,4]] += 1
                
        if (len(self.instance_feature)-self.last_num)>10:
            self.last_num = len(self.instance_feature)
            Log(f"Now the number of instance is: {len(self.instance_feature)}", tag="Open-Instance" )
        if self.vis_gui:
            self.update_vis()
        return   
            
            
    def get_instance_color(self):
        combined_indices = self.combined_indices[self.all_vaild]
        # get the max_id for the voxel_coords
        max_pro_indices = torch.argmax(self.instance_pro_vol[...,:3], dim=-1, keepdim=True)
        max_pro_instance_id = torch.gather(self.instance_id_vol, dim=-1, index=max_pro_indices).squeeze(-1).to(torch.long)
        this_image_instance_id = max_pro_instance_id[combined_indices[:, 0], combined_indices[:, 2], combined_indices[:, 3], combined_indices[:, 4]]
        this_image_instance_count = self.instance_count_vol[combined_indices[:, 0], combined_indices[:, 2], combined_indices[:, 3], combined_indices[:, 4]]
        ok_mask = this_image_instance_count<=self.vox_min_count
        this_image_instance_id[ok_mask] = 0
        unique_labels, _ = torch.unique(this_image_instance_id, return_counts=True)
        unique_labels_count = self.instance_fea_count[unique_labels][:,0]
        labels_to_remove_min_count = unique_labels[unique_labels_count < self.ins_min_count]
        mask = torch.isin(this_image_instance_id, labels_to_remove_min_count)
        this_image_instance_id[mask] = 0
        unique_labels, _ = torch.unique(this_image_instance_id, return_counts=True)
        colors = torch.index_select(self.instance_colors, 0, this_image_instance_id)
        if self.x_coords is None:
            x_coords, y_coords = torch.meshgrid(torch.arange(self.width), torch.arange(self.height))  # x and y are the coordinate grids of width and height
            self.x_coords = x_coords.flatten().cuda()
            self.y_coords = y_coords.flatten().cuda()
        instance_image = torch.zeros((self.height * self.width, 3), dtype=torch.int64).cuda()
        pixel_coords = self.y_coords * self.width + self.x_coords 
        instance_image[pixel_coords[self.all_vaild]] = colors
        self.instance_image = instance_image.view(self.height, self.width, 3).cpu().numpy()

    
        
    
    
    def initializing_check(self):
        '''check the initialized voxels coors and colors'''
        unique_indices = torch.unique(self.combined_indices[self.all_vaild], dim=0)
        new_voxels_mask = self.new_voxel[unique_indices[:,0],unique_indices[:,2],unique_indices[:,3],unique_indices[:,4]]
        
        combined_indices_init = unique_indices[new_voxels_mask]
        colors = torch.utils.dlpack.from_dlpack(self.world.attribute("color").to_dlpack())[combined_indices_init[:,0],combined_indices_init[:,2],
                                                                                                 combined_indices_init[:,3],combined_indices_init[:,4]]/255.0
        points = self.voxel_coords_ok[combined_indices_init[:,1],combined_indices_init[:,2], combined_indices_init[:,3],combined_indices_init[:,4]]
        colors_ok = colors.sum(dim=1) > self.config["Training"]["rgb_boundary_threshold"]
        full_indices = unique_indices[new_voxels_mask][colors_ok]
        self.new_voxel[full_indices[:,0],full_indices[:,2],full_indices[:,3],full_indices[:,4]] = False
        points = points[colors_ok]
        colors = colors[colors_ok]

        # debug
        if len(points)>0:
            pc = o3d.geometry.PointCloud()
            pc.points = o3d.utility.Vector3dVector(points.cpu().numpy())
            pc.colors = o3d.utility.Vector3dVector(colors.cpu().numpy())
            cl, ind = pc.remove_radius_outlier(nb_points=5, radius=0.3)
            points = points.cpu().numpy()[ind]
            colors = colors.cpu().numpy()[ind]
            pc_new = o3d.geometry.PointCloud()
            pc_new.points = o3d.utility.Vector3dVector(points)
            pc_new.colors = o3d.utility.Vector3dVector(colors)
            # o3d.visualization.draw_geometries([pc_new])
            self.all_pc = self.all_pc + pc_new

        # check for keyframe
        self.unregistered_mask[combined_indices_init[:,1],combined_indices_init[:,2],combined_indices_init[:,3],combined_indices_init[:,4]] = True
        # this frame_unregistered
        this_un = self.unregistered_mask[unique_indices[:,1],unique_indices[:,2],unique_indices[:,3],unique_indices[:,4]]
        if this_un.sum()/len(unique_indices) >= self.unregistered_threshold:
            is_keyframe = True
            self.unregistered_mask.fill_(False)
        else:
            is_keyframe = False
        return torch.tensor(points).cuda(), torch.tensor(colors).cuda(), is_keyframe
    
    def reset_unregistered(self):
        self.unregistered_mask.fill_(False)
        
    
    
    
    def get_vis_imgs(self, depth_im=None):
        if depth_im is not None:
            self.depth_im = self.depth_im.float()
            self.depth_im[self.depth_im<0.01] = depth_im[self.depth_im<0.01]
        depth_im = self.depth_im
        if self.vis_gui:
            # project the rgbd image to instance_voxels to get instance color
            self.get_instance_color()
        return self.last_mask_image, self.instance_image, len(self.instance_fea_count)
        
    def adjust_embed_capacity(self):
        if self.world.hashmap().capacity() > self.instance_id_vol.shape[0]:
            print("!"*1000)
            print("new hashmap", self.world.hashmap().capacity())
            delta = self.world.hashmap().capacity() - self.instance_id_vol.shape[0]
            self.instance_id_vol = torch.cat([self.instance_id_vol, torch.zeros(delta, self.block_resolution, self.block_resolution, self.block_resolution, 3).long().to(self.device)], dim=0)
            self.instance_pro_vol = torch.cat([self.instance_pro_vol, torch.zeros(delta, self.block_resolution, self.block_resolution, self.block_resolution, 4).long().to(self.device)], dim=0)
            self.instance_count_vol = torch.cat([self.instance_count_vol, torch.zeros(delta, self.block_resolution, self.block_resolution, self.block_resolution).long().to(self.device)], dim=0)


    def get_all_voxels(self, if_confidence = False):
        buf_indices = self.world.hashmap().active_buf_indices()
        voxel_coords, voxel_indices = self.world.voxel_coordinates_and_flattened_indices(buf_indices)
        o3c.cuda.synchronize()
        # all voxel_coords and voxel_indices that belong to cur_buf_indices block
        voxel_coords, _ = self.world.voxel_coordinates_and_flattened_indices(buf_indices)
        buf_indices = torch.utils.dlpack.from_dlpack(buf_indices.to_dlpack()) # (M,)
        voxel_coords = torch.utils.dlpack.from_dlpack(voxel_coords.to_dlpack()) # (Mx8x8x8,3)
        voxel_coords_ok = voxel_coords.view(-1, self.block_resolution, self.block_resolution, self.block_resolution, 3)  # (M,8,8,8,3)
        
        buf_instance_count_vol = self.instance_count_vol[buf_indices]
        count_mask = buf_instance_count_vol>self.vox_min_count
        points = voxel_coords_ok[count_mask]
        colors = torch.utils.dlpack.from_dlpack(self.world.attribute('color').to_dlpack())[buf_indices]
        pc_colors = colors[count_mask]
        
        buf_instance_pro_vol = self.instance_pro_vol[buf_indices]
        max_pro_indices = torch.argmax(buf_instance_pro_vol[...,:3], dim=-1, keepdim=True)
        buf_instance_id_vol = self.instance_id_vol[buf_indices]
        max_pro_instance_id = torch.gather(buf_instance_id_vol, dim=-1, index=max_pro_indices).squeeze(-1)[count_mask].to(torch.long)
        norm_confidence, confidence_colors = None, None
        if if_confidence:
            confidence  =  torch.max(buf_instance_pro_vol[...,:3], dim=-1).values[count_mask]
            norm_confidence = confidence.float()/self.instance_count_vol[buf_indices][count_mask].float()
            cmap = plt.cm.rainbow
            confidence_colors = cmap(norm_confidence.cpu().numpy())[:, :3] # RGB channels only
        # remove the small label
        unique_labels, counts = torch.unique(max_pro_instance_id, return_counts=True)
        labels_to_remove_min_voxel = unique_labels[counts < self.ins_min_voxel]
        unique_labels_count = self.instance_fea_count[unique_labels][:,0]
        labels_to_remove_min_count = unique_labels[unique_labels_count < self.ins_min_count]
        labels_to_remove = torch.unique(torch.cat((labels_to_remove_min_voxel, labels_to_remove_min_count)))
        mask = torch.isin(max_pro_instance_id, labels_to_remove)
        points = points[~mask]
        pc_colors = pc_colors[~mask]
        max_pro_instance_id = max_pro_instance_id[~mask]
        if if_confidence:
            confidence_colors = confidence_colors[~mask.cpu().numpy()]
        ins_colors = torch.index_select(self.instance_colors, 0, max_pro_instance_id)/255.0
        return points, pc_colors, ins_colors, max_pro_instance_id, confidence_colors, unique_labels, labels_to_remove


    def vis_ply_final(self, vis=False):        
        points, pc_colors, ins_colors, max_pro_instance_id, confidence_colors, unique_labels, labels_to_remove = self.get_all_voxels(if_confidence=True)
        Log(f"Before move the instance is {len(unique_labels)}, after is {len(unique_labels)-len(labels_to_remove)}", tag="Open-Instance")
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(points.cpu().numpy())
        pc.colors = o3d.utility.Vector3dVector(ins_colors.cpu().numpy())
        if vis:
            o3d.visualization.draw_geometries([pc])
        o3d.io.write_point_cloud(f"{self.save_dir}/instance.ply", pc)
        pc.colors = o3d.utility.Vector3dVector(confidence_colors)
        o3d.io.write_point_cloud(f"{self.save_dir}/instance_confidence.ply", pc)
        # save instance pc
        instance_pc = np.hstack((points.cpu().numpy(), max_pro_instance_id.unsqueeze(dim=-1).cpu().numpy(), pc_colors.cpu().numpy()))
        np.save(f"{self.save_dir}/instance_ids.npy", instance_pc)
        
        
    def get_instance_ids(self, points):
        voxels, _, ins_colors, voxels_id, _, _, _ = self.get_all_voxels()
        kdtree = KDTree(voxels.cpu().numpy())
        instance_ids = torch.zeros((len(points))).to(self.device).long()
        for i in range(len(points)):
            # Query KDTree for the nearest valid point
            nearest_idx = kdtree.query(points[i].cpu().numpy(), k=1)[1]
            nearest_instance_id = voxels_id[nearest_idx]
            nearest_point = voxels[nearest_idx]
            distance = torch.norm(points[i] - nearest_point, p=2)
            if distance < self.fill_dis: 
                instance_ids[i] = nearest_instance_id
        colors = torch.index_select(self.instance_colors, 0, instance_ids)/255.0
        return instance_ids, colors
    
    
    
    def get_tsdf_and_weights(self, points):
        # denoise
        cl, ind = self.all_pc.remove_radius_outlier(nb_points=8, radius=0.1)
        all_points = np.asarray(self.all_pc.points)[ind]
        all_colors = np.asarray(self.all_pc.colors)[ind]
        self.all_pc = o3d.geometry.PointCloud()
        self.all_pc.points = o3d.utility.Vector3dVector(all_points)
        self.all_pc.colors = o3d.utility.Vector3dVector(all_colors)
        points_np = points.cpu().numpy()
        # creat a KDTree
        kdtree = KDTree(all_points)
        dist_matrix, _ = kdtree.query(points_np, k=1)
        shortest_dists = torch.from_numpy(dist_matrix).cuda()
        all_mask = shortest_dists > self.voxel_size/2.0
        return all_mask
    
    
        
    def finalize(self):
        mesh = self.get_mesh()
        o3d.io.write_triangle_mesh(f'{self.save_dir}/tsdf_mesh.ply', mesh)
        self.vis_ply_final()
        # save instance caption feature,   N,384
        torch.save(self.instance_feature, f'{self.save_dir}/instance_feature.pt')