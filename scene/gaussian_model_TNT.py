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
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
import torch.nn.functional as F


class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        # self.bound_activation = lambda x: torch.exp(torch.clamp(x, 0, 2))
        # self.factor_activation = lambda x: torch.exp(torch.clamp_min(x, 0))

        # self.bound_activation = lambda x: 3 * torch.sigmoid(x)
        # self.factor_activation = torch.exp

        self.bound_activation = torch.relu
        self.factor_activation = lambda x: torch.exp(torch.clamp_min(x, -2))
        # self.factor_activation = torch.relu

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int, prune_contrib_thres: float = 0.1):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._scaling2 = torch.empty(0)
        self._rotation = torch.empty(0)
        self._omega = torch.empty(0)
        self._opacity = torch.empty(0)
        self._bound = torch.empty(0)
        self._percept_factor = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.xyz_gradient_accum_abs = torch.empty(0)
        self.contrib_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.prune_contrib_thres = prune_contrib_thres
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._scaling2,
            self._rotation,
            self._omega,
            self._bound,
            self._percept_factor,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.xyz_gradient_accum_abs,
            self.contrib_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._scaling2,
        self._rotation, 
        self._omega,
        self._percept_factor,
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        xyz_gradient_accum_abs,
        contrib_accum,
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.xyz_gradient_accum_abs = xyz_gradient_accum_abs
        self.contrib_accum = contrib_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_scaling2(self):
        return self.scaling_activation(self._scaling2)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_omega(self):
        return self._omega

    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_bound(self):
        return self.bound_activation(self._bound)

    @property
    def get_percept_factor(self):
        return self.factor_activation(self._percept_factor)

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_xyz, self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 2) + torch.log(torch.tensor([1.5, 0.5]).cuda().float())
        scales2 = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 2) + torch.log(torch.tensor([1.5, 0.5]).cuda().float())
        rots = torch.rand((fused_point_cloud.shape[0], 4), device="cuda")

        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
        self._omega = torch.ones_like(opacities) * 0.017 * 45

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._scaling2 = nn.Parameter(scales2.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self._bound = nn.Parameter(torch.ones_like(self._opacity) * 2)
        # self._percept_factor = nn.Parameter(torch.log(torch.ones_like(self._opacity) * 1))
        self._percept_factor = nn.Parameter(torch.zeros_like(self._opacity))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum_abs = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.contrib_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._scaling2], 'lr': training_args.scaling_lr, "name": "scaling2"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            {'params': [self._bound], 'lr': training_args.scaling_lr, "name": "bound"},
            {'params': [self._percept_factor], 'lr': training_args.scaling_lr, "name": "percept_factor"},
            {'params': [self._omega], 'lr': training_args.rotation_lr, "name": "omega"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._scaling2.shape[1]):
            l.append('scale2_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        for i in range(self._omega.shape[1]):
            l.append('omega_{}'.format(i))
        l.append('bound')
        l.append('percept_factor')
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        scale2 = self._scaling2.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        omega = self._omega.detach().cpu().numpy()
        bound = self._bound.detach().cpu().numpy()
        percept_factor = self._percept_factor.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, scale2, rotation, omega, bound, percept_factor), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

        # scales1_new = self.scaling_inverse_activation(self.scaling_activation(self._scaling) / 2)
        # optimizable_tensors = self.replace_tensor_to_optimizer(scales1_new, "scaling")
        # self._scaling = optimizable_tensors["scaling"]

        # scales2_new = self.scaling_inverse_activation(self.scaling_activation(self._scaling2) / 2)
        # optimizable_tensors = self.replace_tensor_to_optimizer(scales2_new, "scaling2")
        # self._scaling2 = optimizable_tensors["scaling2"]

        # percept_factor = torch.ones_like(self._percept_factor)
        # optimizable_tensors = self.replace_tensor_to_optimizer(percept_factor, "percept_factor")
        # self._percept_factor = optimizable_tensors["percept_factor"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale2_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales2 = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales2[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        omega_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("omega")]
        omega_names = sorted(omega_names, key = lambda x: int(x.split('_')[-1]))
        omegas = np.zeros((xyz.shape[0], len(omega_names)))
        for idx, attr_name in enumerate(omega_names):
            omegas[:, idx] = np.asarray(plydata.elements[0][attr_name])

        bound = np.asarray(plydata.elements[0]["bound"])[..., np.newaxis]
        percept_factor = np.asarray(plydata.elements[0]["percept_factor"])[..., np.newaxis]
        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling2 = nn.Parameter(torch.tensor(scales2, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self._omega = nn.Parameter(torch.tensor(omegas, dtype=torch.float, device="cuda").requires_grad_(True))
        self._bound = nn.Parameter(torch.tensor(bound, dtype=torch.float, device="cuda").requires_grad_(True))
        self._percept_factor = nn.Parameter(torch.tensor(percept_factor, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._scaling2 = optimizable_tensors["scaling2"]
        self._rotation = optimizable_tensors["rotation"]
        self._omega = optimizable_tensors["omega"]
        self._bound = optimizable_tensors["bound"]
        self._percept_factor = optimizable_tensors["percept_factor"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.xyz_gradient_accum_abs = self.xyz_gradient_accum_abs[valid_points_mask]
        self.contrib_accum = self.contrib_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_scaling2, new_rotation, new_omega, new_bound, new_percept_factor):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "scaling2" : new_scaling2,
        "omega": new_omega,
        "bound": new_bound,
        "percept_factor": new_percept_factor,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._scaling2 = optimizable_tensors["scaling2"]
        self._rotation = optimizable_tensors["rotation"]
        self._omega = optimizable_tensors["omega"]
        self._bound = optimizable_tensors["bound"]
        self._percept_factor = optimizable_tensors["percept_factor"]


        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum_abs = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.contrib_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def get_scaling_max_max(self):
        # return torch.max(torch.min(self.get_scaling, dim=1).values, torch.min(self.get_scaling, dim=1).values)
        with torch.no_grad():
            scaling_1 = self.get_scaling
            scaling_2 = self.get_scaling2
            sin_omega = abs(torch.sin(self._omega))
            cos_omega = abs(torch.cos(self._omega))
            scaling_2_x, scaling_2_y = torch.split(scaling_2, 1, dim=-1)
            scaling_2_0 = torch.max(cos_omega * scaling_2_x, sin_omega * scaling_2_y)
            scaling_2_1 = torch.max(sin_omega * scaling_2_x, cos_omega * scaling_2_y)
            scaling_2_rot = torch.cat([scaling_2_0, scaling_2_1], dim=-1)
            # print(scaling_1.shape, scaling_2_rot.shape)
            min_scaling = torch.max(scaling_1, scaling_2_rot)
            return torch.max(min_scaling, dim=-1).values
        
    def get_scaling_max_min(self):
        with torch.no_grad():
            scaling_1 = self.get_scaling
            scaling_2 = self.get_scaling2
            sin_omega = abs(torch.sin(self._omega))
            cos_omega = abs(torch.cos(self._omega))
            scaling_2_x, scaling_2_y = torch.split(scaling_2, 1, dim=-1)
            scaling_2_0 = torch.max(cos_omega * scaling_2_x, sin_omega * scaling_2_y)
            scaling_2_1 = torch.max(sin_omega * scaling_2_x, cos_omega * scaling_2_y)
            scaling_2_rot = torch.cat([scaling_2_0, scaling_2_1], dim=-1)
            # print(scaling_1.shape, scaling_2_rot.shape)
            min_scaling = torch.max(scaling_1, scaling_2_rot)
            return torch.min(min_scaling, dim=-1).values

    def densify_and_split(self, grads, grad_threshold,  grads_abs, grad_abs_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        padded_grad_abs = torch.zeros((n_init_points), device="cuda")
        padded_grad_abs[:grads_abs.shape[0]] = grads_abs.squeeze()
        selected_pts_mask_abs = torch.where(padded_grad_abs >= grad_abs_threshold, True, False)
        selected_pts_mask = torch.logical_or(selected_pts_mask, selected_pts_mask_abs)
        # selected_pts_mask = torch.logical_and(selected_pts_mask,
        #                                       torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)
        
        # selected_pts_mask = torch.logical_and(
        #     selected_pts_mask,
        #     torch.max(torch.max(self.get_scaling, self.get_scaling2), dim=1).values > self.percent_dense*scene_extent)


        # stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        # stds = torch.cat([stds, 0 * torch.ones_like(stds[:,:1])], dim=-1)
        # means = torch.zeros_like(stds)
        # samples = torch.normal(mean=means, std=stds)
        # rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        scaling_1 = self.get_scaling[selected_pts_mask]
        scaling_2 = self.get_scaling2[selected_pts_mask]
        sin_omega = abs(torch.sin(self._omega[selected_pts_mask]))
        cos_omega = abs(torch.cos(self._omega[selected_pts_mask]))
        scaling_2_x, scaling_2_y = torch.split(scaling_2, 1, dim=-1)
        scaling_2_0 = torch.max(cos_omega * scaling_2_x, sin_omega * scaling_2_y)
        scaling_2_1 = torch.max(sin_omega * scaling_2_x, cos_omega * scaling_2_y)
        scaling_2_rot = torch.cat([scaling_2_0, scaling_2_1], dim=-1)
        stds = torch.min(scaling_1, scaling_2_rot)
        stds = torch.cat([stds[:,:1], torch.zeros_like(stds[:,:2])], dim=-1)
        dis = torch.sqrt(self.get_bound[selected_pts_mask]) / 2
        samples = torch.cat([stds * -1 * dis, stds * dis], dim=0)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)

        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask] / 1.6).repeat(N,1)
        new_scaling2 = self.scaling_inverse_activation(self.get_scaling2[selected_pts_mask] / 1.6).repeat(N,1)
        # new_scaling = torch.flip(new_scaling, dims=[-1] )
        # new_scaling2 = torch.flip(new_scaling2 , dims=[-1] )
        new_omega = self._omega[selected_pts_mask].repeat(N,1)
        
        # new_rotation = F.normalize(get_new_rot(F.normalize(self._rotation[selected_pts_mask]))).repeat(N,1)
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_bound = self._bound[selected_pts_mask].repeat(N,1)
        new_percept_factor = self._percept_factor[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_scaling2, new_rotation, new_omega, new_bound, new_percept_factor)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold,  grads_abs, grad_abs_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask_abs = torch.where(torch.norm(grads_abs, dim=-1) >= grad_abs_threshold, True, False)
        selected_pts_mask = torch.logical_or(selected_pts_mask, selected_pts_mask_abs)
        # selected_pts_mask = torch.logical_and(selected_pts_mask,
        #                                       torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        # print(scene_extent, self.percent_dense)
        # exit()
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(torch.max(self.get_scaling, self.get_scaling2), dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_scaling2 = self._scaling2[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_omega = self._omega[selected_pts_mask]
        new_bound = self._bound[selected_pts_mask]
        new_percept_factor = self._percept_factor[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_scaling2, new_rotation, new_omega, new_bound, new_percept_factor)

    def densify_and_prune(self, ratio, min_opacity, extent, need_prune=False):
        if need_prune:
            contrib = self.contrib_accum / self.denom
            contrib[contrib.isnan()] = 0.0
            prune_mask = (self.get_opacity < min_opacity).squeeze()
            contrib_mask = (contrib < self.prune_contrib_thres).squeeze()
            prune_mask = torch.logical_or(prune_mask, contrib_mask)
            self.prune_points(prune_mask)

        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        grads_abs = self.xyz_gradient_accum_abs / self.denom
        grads_abs[grads_abs.isnan()] = 0.0

        grads = grads / torch.max(grads)
        
        grads_div = grads_abs / (grads + 1e-8)
        grads_div = grads_div / torch.max(grads_div)
        
        Q_grad = torch.quantile(grads.reshape(-1), 1 - ratio * 0.5)
        Q_div = torch.quantile(grads_div.reshape(-1), 1 - ratio * 0.5)

        # self.densify_and_clone(grads, Q_grad, grads_div, Q_div, extent)
        self.densify_and_split(grads, Q_grad, grads_div, Q_div, extent)



        torch.cuda.empty_cache()


    def add_densification_stats(self, viewspace_point_tensor, contrib_tensor, update_filter):
        mask = torch.ones_like(viewspace_point_tensor[update_filter])
        mask[:, -1] = 0
        self.xyz_gradient_accum[update_filter] += torch.norm(
            abs(viewspace_point_tensor.grad[update_filter] * mask), 
            dim=-1, keepdim=True
        )
        
        mask[:, -1] = 1
        mask[:, :-1] = 0
        
        self.xyz_gradient_accum_abs[update_filter] += torch.norm(
            viewspace_point_tensor.grad[update_filter] * mask, 
            dim=-1, keepdim=True
        )
        # print(contrib_tensor.shape, self.contrib_accum.shape)
        self.contrib_accum[update_filter] += contrib_tensor[:, None][update_filter].detach()
        self.denom[update_filter] += 1