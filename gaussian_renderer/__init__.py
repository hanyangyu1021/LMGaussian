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
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from utils.graphics_utils import fov2focal, getcam_center_tensor, getWorld2View2, getProjectionMatrix
import msplat


def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0):
    """
    Render the scene. 
    
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
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    scales, opacity = pc.get_scaling_n_opacity_with_3D_filter
    rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = pc.get_features
    colors_precomp = None

    rendered_image, radii, rendered_depth, rendered_middepth, rendered_alpha, rendered_normal, depth_distortion = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)



    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "mask": rendered_alpha,
            "depth": rendered_depth,
            "middepth": rendered_middepth,
            "viewspace_points": means2D,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "normal":rendered_normal,
            "depth_distortion": depth_distortion,
            }

def render_hard(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0):
    """
    Hard loss
    fix the opacity==0.95,  scales, rotations to force network to predict true depth
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
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    scales, opacity = pc.get_scaling_n_opacity_with_3D_filter
    scales = scales.detach()
    opacity = opacity.detach()
    opacity = torch.ones(pc.get_xyz.shape[0], 1, device=pc.get_xyz.device) * 0.95
    rotations = pc.get_rotation.detach()

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = pc.get_features
    colors_precomp = None

    rendered_image, radii, rendered_depth, rendered_middepth, rendered_alpha, rendered_normal, depth_distortion = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)



    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "mask": rendered_alpha,
            "depth": rendered_depth,
            "middepth": rendered_middepth,
            "viewspace_points": means2D,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "normal":rendered_normal,
            "depth_distortion": depth_distortion,
            }

def render_soft(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0):
    """
    Hard loss
    fix the xyz, scales, rotations to force network to predict true depth
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
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz.detach()
    means2D = screenspace_points

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    scales, opacity = pc.get_scaling_n_opacity_with_3D_filter
    scales = scales.detach()
    rotations = pc.get_rotation.detach()

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = pc.get_features
    colors_precomp = None

    rendered_image, radii, rendered_depth, rendered_middepth, rendered_alpha, rendered_normal, depth_distortion = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)


    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "mask": rendered_alpha,
            "depth": rendered_depth,
            "middepth": rendered_middepth,
            "viewspace_points": means2D,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "normal":rendered_normal,
            "depth_distortion": depth_distortion,
            }

def render_msplat_test(viewpoint_camera, pc : GaussianModel, original_pose : torch.Tensor, extr_opt = None): 
    """
    Render the scene. Optimize gaussian and camera parameters at the same time.
    """
    fx = fov2focal(viewpoint_camera.FoVx, viewpoint_camera.image_width)
    fy = fov2focal(viewpoint_camera.FoVy, viewpoint_camera.image_height)
    intr = torch.Tensor([fx, fy, viewpoint_camera.image_width / 2, viewpoint_camera.image_height / 2]).cuda().float()
    torch.autograd.set_detect_anomaly(True)
    if extr_opt == None:
        extr = original_pose.cuda()
    else:
        extr = extr_opt

    shs_view = pc.get_features.detach().transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
    R_opt = torch.transpose(extr[:,:3],0,1)
    T_opt = extr[:,3]
    camera_center = getcam_center_tensor(R_opt, T_opt, torch.tensor([0., 0., 0.]), 1.0).cuda()


    direction = (pc.get_xyz.detach() -
                    camera_center.repeat(pc.get_xyz.shape[0], 1))
    direction = direction / direction.norm(dim=1, keepdim=True)
    sh2rgb = msplat.compute_sh(shs_view, direction)
    rgb = torch.clamp_min(sh2rgb + 0.5, 0.0)
    (uv, depth) = msplat.project_point(
            pc.get_xyz.detach(),
            intr,
            extr,
            viewpoint_camera.image_width, viewpoint_camera.image_height)

    visible = depth != 0

    # compute cov3d
    cov3d = msplat.compute_cov3d(pc.get_scaling.detach(), pc.get_rotation.detach(), visible)

    # ewa project
    (conic, radius, tiles_touched) = msplat.ewa_project(
        pc.get_xyz.detach(),
        cov3d,
        intr,
        extr,
        uv,
        viewpoint_camera.image_width,
        viewpoint_camera.image_height,
        visible
    )

    # sort
    (gaussian_ids_sorted, tile_range) = msplat.sort_gaussian(
        uv, depth, viewpoint_camera.image_width, viewpoint_camera.image_height, radius, tiles_touched
    )

    ndc = torch.zeros_like(uv, requires_grad=True)
    try:
        ndc.retain_grad()
    except:
        raise ValueError("ndc does not have grad")

    # alpha blending
    rendered_features = msplat.alpha_blending(
        uv, conic, pc.get_opacity.detach(), rgb,
        gaussian_ids_sorted, tile_range, 0, viewpoint_camera.image_width, viewpoint_camera.image_height, ndc
    )

    return {"render": rendered_features,
            "viewspace_points": ndc,
            "visibility_filter": radius > 0,
            "radii": radius
            }

def render_point(viewpoint_camera, pc : GaussianModel, camera_center, world_view_transform, full_proj_transform, \
                 pipe, bg_color : torch.Tensor, scaling_modifier = 1.0):
    """
    Render the scene. 
    
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
        viewmatrix=world_view_transform,
        projmatrix=full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    scales, opacity = pc.get_scaling_n_opacity_with_3D_filter
    rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = pc.get_features
    colors_precomp = None

    rendered_image, radii, rendered_depth, rendered_middepth, rendered_alpha, rendered_normal, depth_distortion = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)



    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "mask": rendered_alpha,
            "depth": rendered_depth,
            "middepth": rendered_middepth,
            "viewspace_points": means2D,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "normal":rendered_normal,
            "depth_distortion": depth_distortion,
            }

def render_test(viewpoint_camera, pc : GaussianModel, extr_opt, \
                pipe, bg_color : torch.Tensor, scaling_modifier = 1.0):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    test_R = torch.transpose(extr_opt[:3,:3], 0, 1).cpu().numpy()
    test_T = extr_opt[:3, 3].cpu().numpy()
    world_view_transform = torch.tensor(getWorld2View2(test_R, test_T, \
                        viewpoint_camera.trans, viewpoint_camera.scale)).transpose(0, 1).cuda()
    projection_matrix = getProjectionMatrix(viewpoint_camera.znear, viewpoint_camera.zfar, viewpoint_camera.FoVx, viewpoint_camera.FoVy).transpose(0,1).cuda()
    full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)

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
        viewmatrix=world_view_transform,
        projmatrix=full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points

    scales = None
    rotations = None
    cov3D_precomp = None
    scales, opacity = pc.get_scaling_n_opacity_with_3D_filter
    rotations = pc.get_rotation

    shs = pc.get_features
    colors_precomp = None

    rendered_image, radii, rendered_depth, rendered_middepth, rendered_alpha, rendered_normal, depth_distortion = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    return {"render": rendered_image,
            "mask": rendered_alpha,
            "depth": rendered_depth,
            "middepth": rendered_middepth,
            "viewspace_points": means2D,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "normal":rendered_normal,
            "depth_distortion": depth_distortion,
            }


# integration is adopted from GOF for marching tetrahedra https://github.com/autonomousvision/gaussian-opacity-fields/blob/main/gaussian_renderer/__init__.py
def integrate(points3D, viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    """
    integrate Gaussians to the points, we also render the image for visual comparison. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

        
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity_with_3D_filter

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling_with_3D_filter
        rotations = pc.get_rotation

    depth_plane_precomp = None

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            # # we local direction
            # cam_pos_local = view2gaussian_precomp[:, 3, :3]
            # cam_pos_local_scaled = cam_pos_local / scales
            # dir_pp = -cam_pos_local_scaled
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, alpha_integrated, color_integrated, point_coordinate, point_sdf, radii = rasterizer.integrate(
        points3D = points3D,
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp,
        view2gaussian_precomp=depth_plane_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "alpha_integrated": alpha_integrated,
            "color_integrated": color_integrated,
            "point_coordinate": point_coordinate,
            "point_sdf": point_sdf,
            "visibility_filter" : radii > 0,
            "radii": radii}