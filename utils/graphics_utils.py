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
import numpy as np
from typing import NamedTuple
import cv2
import os
import re

class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array

def geom_transform_points(points, transf_matrix):
    P, _ = points.shape
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))

    denom = points_out[..., 3:] + 0.0000001
    return (points_out[..., :3] / denom).squeeze(dim=0)

def getWorld2View(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    """ get world 2 camera matrix

    Args:
        R (_type_): c2w rotation
        t (_type_): w2c camera center
        translate (_type_, optional): _description_. Defaults to np.array([.0, .0, .0]).
        scale (float, optional): _description_. Defaults to 1.0.

    Returns:
        _type_: _description_
    """
    # compose w2c matrix
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    # invert to get c2w
    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    # get the final w2c matrix
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))


# the following functions depths_double_to_points and depth_double_to_normal are adopted from https://github.com/hugoycj/2dgs-gaustudio/blob/main/utils/graphics_utils.py
def depths_double_to_points(view, depthmap1, depthmap2):
    W, H = view.image_width, view.image_height
    fx = W / (2 * math.tan(view.FoVx / 2.))
    fy = H / (2 * math.tan(view.FoVy / 2.))
    intrins = torch.tensor(
        [[fx, 0., W/2.],
        [0., fy, H/2.],
        [0., 0., 1.0]]
    ).float().cuda()
    grid_x, grid_y = torch.meshgrid(torch.arange(W)+0.5, torch.arange(H)+0.5, indexing='xy')
    points = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1).reshape(-1, 3).float().cuda()
    rays_d = points @ intrins.inverse().T
    # rays_o = torch.zeros(3,dtype=torch.float32,device="cuda")
    # rays_o = c2w[:3,3]
    points1 = depthmap1.reshape(-1, 1) * rays_d
    points2 = depthmap2.reshape(-1, 1) * rays_d
    return points1, points2



def depth_double_to_normal(view, depth1, depth2):
    points1, points2 = depths_double_to_points(view, depth1, depth2)
    points = torch.stack([points1, points2],dim=0).reshape(2, *depth1.shape[1:], 3)
    output = torch.zeros_like(points)
    dx = points[:,2:, 1:-1] - points[:,:-2, 1:-1]
    dy = points[:,1:-1, 2:] - points[:,1:-1, :-2]
    normal_map = torch.nn.functional.normalize(torch.cross(dx, dy, dim=-1), dim=-1)
    output[:,1:-1, 1:-1, :] = normal_map
    return output, points

def bilinear_sampler(img, coords, mask=False):
    """ Wrapper for grid_sample, uses pixel coordinates """
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1,1], dim=-1)
    xgrid = 2*xgrid/(W-1) - 1
    ygrid = 2*ygrid/(H-1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = torch.nn.functional.grid_sample(img, grid, align_corners=True)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img


# project the reference point cloud into the source view, then project back
#extrinsics here refers c2w
def reproject_with_depth(depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src):
    batch, height, width = depth_ref.shape
    
    ## step1. project reference pixels to the source view
    # reference view x, y
    y_ref, x_ref = torch.meshgrid(torch.arange(0, height).to(depth_ref.device), torch.arange(0, width).to(depth_ref.device))
    x_ref = x_ref.unsqueeze(0).repeat(batch,  1, 1)
    y_ref = y_ref.unsqueeze(0).repeat(batch,  1, 1)
    x_ref, y_ref = x_ref.reshape(batch, -1), y_ref.reshape(batch, -1)
    # reference 3D space

    A = torch.inverse(intrinsics_ref)
    B = torch.stack((x_ref, y_ref, torch.ones_like(x_ref).to(x_ref.device)), dim=1) * depth_ref.reshape(batch, 1, -1)
    xyz_ref = torch.matmul(A, B)

    # source 3D space
    xyz_src = torch.matmul(torch.matmul(torch.inverse(extrinsics_src), extrinsics_ref),
                        torch.cat((xyz_ref, torch.ones_like(x_ref).to(x_ref.device).unsqueeze(1)), dim=1))[:, :3]
    # source view x, y
    K_xyz_src = torch.matmul(intrinsics_src, xyz_src)
    xy_src = K_xyz_src[:, :2] / K_xyz_src[:, 2:3]

    ## step2. reproject the source view points with source view depth estimation
    # find the depth estimation of the source view
    x_src = xy_src[:, 0].reshape([batch, height, width]).float()
    y_src = xy_src[:, 1].reshape([batch, height, width]).float()

    # print(x_src, y_src)
    sampled_depth_src = bilinear_sampler(depth_src.view(batch, 1, height, width), torch.stack((x_src, y_src), dim=-1).view(batch, height, width, 2))

    # source 3D space
    # NOTE that we should use sampled source-view depth_here to project back
    xyz_src = torch.matmul(torch.inverse(intrinsics_src),
                        torch.cat((xy_src, torch.ones_like(x_ref).to(x_ref.device).unsqueeze(1)), dim=1) * sampled_depth_src.reshape(batch, 1, -1))
    # reference 3D space
    xyz_reprojected = torch.matmul(torch.matmul(torch.inverse(extrinsics_ref), extrinsics_src),
                                torch.cat((xyz_src, torch.ones_like(x_ref).to(x_ref.device).unsqueeze(1)), dim=1))[:, :3]
    # source view x, y, depth
    depth_reprojected = xyz_reprojected[:, 2].reshape([batch, height, width]).float()
    K_xyz_reprojected = torch.matmul(intrinsics_ref, xyz_reprojected)
    xy_reprojected = K_xyz_reprojected[:, :2] / K_xyz_reprojected[:, 2:3]
    x_reprojected = xy_reprojected[:, 0].reshape([batch, height, width]).float()
    y_reprojected = xy_reprojected[:, 1].reshape([batch, height, width]).float()

    return depth_reprojected, x_reprojected, y_reprojected, x_src, y_src

def check_geometric_consistency(depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src, thre1=1, thre2=0.01):
    batch, height, width = depth_ref.shape
    y_ref, x_ref = torch.meshgrid(torch.arange(0, height).to(depth_ref.device), torch.arange(0, width).to(depth_ref.device))
    x_ref = x_ref.unsqueeze(0).repeat(batch,  1, 1)
    y_ref = y_ref.unsqueeze(0).repeat(batch,  1, 1)
    inputs = [depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src]
    outputs = reproject_with_depth(*inputs)
    depth_reprojected, x2d_reprojected, y2d_reprojected, x2d_src, y2d_src = outputs
    # check |p_reproj-p_1| < 1
    dist = torch.sqrt((x2d_reprojected - x_ref) ** 2 + (y2d_reprojected - y_ref) ** 2)

    # check |d_reproj-d_1| / d_1 < 0.01
    depth_diff = torch.abs(depth_reprojected - depth_ref)
    relative_depth_diff = depth_diff / depth_ref

    mask = torch.logical_and(dist < thre1, relative_depth_diff < thre2)
    depth_reprojected[~mask] = 0

    return mask, depth_reprojected, x2d_src, y2d_src, relative_depth_diff

def vis_surface_normal_inverse(normal: torch.tensor) -> torch.tensor:
    """
    Visualize surface normal. Transfer surface normal value from [0, 255] to [-1, 1]
    Aargs:
        normal (torch.tensor, [h, w, 3]): surface normal
    """
    normal = normal.to(torch.float32)
    normal_vis = (normal / 127) - 1
    return normal_vis

def vis_surface_normal(normal: torch.tensor) -> np.array:
    """
    Visualize surface normal. Transfer surface normal value from [-1, 1] to [0, 255]
    Aargs:
        normal (torch.tensor, [h, w, 3]): surface normal
        mask (torch.tensor, [h, w]): valid masks
    """
    normal = normal.cpu().numpy().squeeze()
    n_img_L2 = np.sqrt(np.sum(normal ** 2, axis=2, keepdims=True))
    n_img_norm = normal / (n_img_L2 + 1e-8)
    normal_vis = n_img_norm * 127
    normal_vis += 128
    normal_vis = normal_vis.astype(np.uint8)
    return normal_vis

from plyfile import PlyData
from pytorch3d.structures import Pointclouds
import scipy.spatial.transform as transform

def load_pointcloud(DATA_DIR, device):
    obj_filename = os.path.join(DATA_DIR, "train/points3d.ply")
    # Load point cloud
    scale = 100
    plydata = PlyData.read(obj_filename)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T * scale
    verts = torch.tensor(positions).to(torch.float32).to(device)
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    rgb = torch.tensor(colors).to(torch.float32).to(device)
    point_cloud = Pointclouds(points=[verts], features=[rgb])
    return point_cloud


def interpolate_camera_poses(camera_poses, num_virtual_poses):
    virtual_poses_R = []
    virtual_poses_T = []
    virtual_camera_center = []
    virtual_world_view_transform = []
    virtual_full_proj_transform = []
    for i in range(len(camera_poses)-1):
        pose1_index = i
        pose2_index = i + 1

        pose1 = camera_poses[pose1_index].R
        pose2 = camera_poses[pose2_index].R 
        t1 = camera_poses[pose1_index].T
        t2 = camera_poses[pose2_index].T

        r1_2 = transform.Rotation.from_matrix([pose1,pose2])

        for j in range(num_virtual_poses):
            t = j / num_virtual_poses
            interpolated_r = transform.Slerp([0, 1], r1_2)(t).as_matrix()
            interpolated_t = t1 * (1 - t) + t2 * t
            virtual_poses_R.append(interpolated_r)
            virtual_poses_T.append(interpolated_t)

            world_view_transform = torch.tensor(getWorld2View2(interpolated_r, interpolated_t, \
                        camera_poses[pose1_index].trans, camera_poses[pose1_index].scale)).transpose(0, 1).cuda()
            projection_matrix = getProjectionMatrix(camera_poses[pose1_index].znear, camera_poses[pose1_index].zfar, camera_poses[pose1_index].FoVx, camera_poses[pose1_index].FoVy).transpose(0,1).cuda()
            full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
            camera_center = world_view_transform.inverse()[3, :3]

            virtual_camera_center.append(camera_center)
            virtual_full_proj_transform.append(full_proj_transform)
            virtual_world_view_transform.append(world_view_transform)


    return virtual_poses_R, virtual_poses_T, virtual_camera_center, virtual_world_view_transform, virtual_full_proj_transform


def extract_number(filename):
    match = re.search(r'\d+', filename)
    if match:
        return int(match.group())
    else:
        return 0
    
def get_dis_from_ts(T, all_T):
    return torch.sort(torch.sqrt(torch.sum((T - all_T) ** 2, dim=-1)))[0]

def get_numeric_value(file_name):
    numeric_part = ''.join(filter(str.isdigit, file_name))
    if numeric_part:
        return int(numeric_part)
    else:
        return 0
    
def vec2skew(v):
    """
    :param v:  (3, ) torch tensor
    :return:   (3, 3)
    """
    zero = torch.zeros(1, dtype=torch.float32, device=v.device)
    skew_v0 = torch.cat([ zero,    -v[2:3],   v[1:2]])  # (3, 1)
    skew_v1 = torch.cat([ v[2:3],   zero,    -v[0:1]])
    skew_v2 = torch.cat([-v[1:2],   v[0:1],   zero])
    skew_v = torch.stack([skew_v0, skew_v1, skew_v2], dim=0)  # (3, 3)
    return skew_v  # (3, 3)


def Exp(r):
    """so(3) vector to SO(3) matrix
    :param r: (3, ) axis-angle, torch tensor
    :return:  (3, 3)
    """
    skew_r = vec2skew(r)  # (3, 3)
    norm_r = r.norm() + 1e-15
    eye = torch.eye(3, dtype=torch.float32, device=r.device)
    R = eye + (torch.sin(norm_r) / norm_r) * skew_r + ((1 - torch.cos(norm_r)) / norm_r**2) * (skew_r @ skew_r)
    return R

def make_c2w(r, t):
    """
    :param r:  (3, ) axis-angle             torch tensor
    :param t:  (3, ) translation vector     torch tensor
    :return:   (4, 4)
    """
    R = Exp(r)  # (3, 3)
    c2w = torch.cat([R, t.unsqueeze(1)], dim=1)  # (3, 4)
    c2w = convert3x4_4x4(c2w)  # (4, 4)
    return c2w
    
def convert3x4_4x4(input):
    """
    :param input:  (N, 3, 4) or (3, 4) torch or np
    :return:       (N, 4, 4) or (4, 4) torch or np
    """
    if torch.is_tensor(input):
        if len(input.shape) == 3:
            output = torch.cat([input, torch.zeros_like(input[:, 0:1])], dim=1)  # (N, 4, 4)
            output[:, 3, 3] = 1.0
        else:
            output = torch.cat([input, torch.tensor([[0,0,0,1]], dtype=input.dtype, device=input.device)], dim=0)  # (4, 4)
    else:
        if len(input.shape) == 3:
            output = np.concatenate([input, np.zeros_like(input[:, 0:1])], axis=1)  # (N, 4, 4)
            output[:, 3, 3] = 1.0
        else:
            output = np.concatenate([input, np.array([[0,0,0,1]], dtype=input.dtype)], axis=0)  # (4, 4)
            output[3, 3] = 1.0
    return output


def getcam_center_tensor(R, t, translate=torch.tensor([.0, .0, .0]), scale=1.0):
    Rt = torch.zeros((4, 4), dtype=torch.float32)
    Rt[:3, :3] = R.transpose(0, 1)
    Rt[:3, 3] = t.squeeze()
    Rt[3, 3] = 1.0

    C2W = torch.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    return cam_center



