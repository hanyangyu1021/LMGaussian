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

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, pearson_depth_loss, local_pearson_loss
from utils.graphics_utils import vis_surface_normal_inverse,vis_surface_normal, get_dis_from_ts, interpolate_camera_poses
from gaussian_renderer import render, render_point, network_gui
import sys
import cv2
from lpipsPyTorch import lpips
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
import numpy as np
from utils.image_utils import psnr
from utils.graphics_utils import depth_double_to_normal, extract_number
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams360
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
from PIL import Image
from scene.cameras import Camera
from matplotlib import pyplot as plt
from utils.depth_utils import depth_to_normal
from utils.vis_utils import apply_depth_colormap
from torch import nn
from cldm.model import create_model, load_state_dict
from minlora import add_lora, LoRAParametrization
from cldm.ddim_hacked import DDIMSampler
from PIL import Image
import numpy as np
from utils.diff_utils import process
from torchvision.transforms import ToPILImage, ToTensor, CenterCrop
import torchvision.transforms as transforms
import pytorch_lightning as pl
from functools import partial
import random


# function L1_loss_appearance is fork from GOF https://github.com/autonomousvision/gaussian-opacity-fields/blob/main/train.py
def L1_loss_appearance(image, gt_image, gaussians, view_idx, return_transformed_image=False):
    appearance_embedding = gaussians.get_apperance_embedding(view_idx)
    # center crop the image
    origH, origW = image.shape[1:]
    H = origH // 32 * 32
    W = origW // 32 * 32
    left = origW // 2 - W // 2
    top = origH // 2 - H // 2
    crop_image = image[:, top:top+H, left:left+W]
    crop_gt_image = gt_image[:, top:top+H, left:left+W]
    
    # down sample the image
    crop_image_down = torch.nn.functional.interpolate(crop_image[None], size=(H//32, W//32), mode="bilinear", align_corners=True)[0]
    
    crop_image_down = torch.cat([crop_image_down, appearance_embedding[None].repeat(H//32, W//32, 1).permute(2, 0, 1)], dim=0)[None]
    mapping_image = gaussians.appearance_network(crop_image_down)
    transformed_image = mapping_image * crop_image
    if not return_transformed_image:
        return l1_loss(transformed_image, crop_gt_image)
    else:
        transformed_image = torch.nn.functional.interpolate(transformed_image, size=(origH, origW), mode="bilinear", align_corners=True)[0]
        return transformed_image

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, \
              controlnet, ddim_sampler, iter_num):
    set_seed(10)
    first_iter = 0
    iter_rounds = []
    for i in range(iter_num):
        iter_rounds.append(checkpoint_iterations[0]+ 6000*i)
    dataset.model_path, _ = os.path.split(checkpoint) # the first stage 3dgs
    tb_writer = prepare_output_and_logger(dataset)
    visualize = True
    opt.opt_train_depth = False
    opt.opt_train_normal = False
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, opt.opt_train_depth, opt.opt_train_normal, load_iteration = checkpoint_iterations[0], gap = pipe.interval)
    C, H, W = scene.train_cameras[1][1].original_image.shape
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")    
    trainCameras = scene.getTrainCameras().copy()
    gaussians.compute_3D_filter(cameras=trainCameras)

    # distance-aware weight 
    train_T_list = []
    gt_images = []
    max_cam_dis = 0.
    for train_camera in scene.getTrainCameras():
        T = torch.from_numpy(train_camera.T).to(train_camera.data_device)
        train_T_list.append(T)
        gt_images.append(train_camera.original_image)

    sparse_num = len(train_T_list)
    for T in train_T_list:
        distances = get_dis_from_ts(T,  torch.stack(train_T_list))
        max_cam_dis = max(max_cam_dis, distances[sparse_num - 2].cpu().item())
    max_cam_dis *= 1.2

    transform = transforms.Compose([
        ToTensor(),
        CenterCrop((train_camera.image_height, train_camera.image_width))
        ])
    

    standard_cam = scene.getTrainCameras()[0]
    save_path = dataset.model_path + '/diff/' + str(first_iter) + '_gs/'
    save_path_diff = dataset.model_path + '/diff/' + str(first_iter) + '_diff/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.exists(save_path_diff):
        os.makedirs(save_path_diff)
    num_virtual_poses = 10
    diff_outputs = []
    virtual_cameras_R,  virtual_cameras_t, virtual_camera_center, virtual_world_view_transform, virtual_full_proj_transform \
        = interpolate_camera_poses(scene.train_cameras[1.0], num_virtual_poses)
    if W >=1600:
        trans = transforms.Resize([H, W], antialias=True)
    ##---------------------------first time should do-------------------------------#
    for num in range(len(virtual_cameras_R)):
        camera_center = virtual_camera_center[num]
        world_view_transform = virtual_world_view_transform[num]
        full_proj_transform = virtual_full_proj_transform[num]
        render_pkg_point = render_point(standard_cam, gaussians, \
                    camera_center, world_view_transform, full_proj_transform, \
                    pipe, background) 
        rendered_image = render_pkg_point["render"]
        if rendered_image.shape[2] >= 1600:
            rendered_image = trans(rendered_image)
        image_np = rendered_image.detach().cpu().numpy() * 255
        image_np = np.uint8(np.transpose(np.clip(image_np, 0, 255), (1, 2, 0)))
        if visualize:
            image = Image.fromarray(image_np)
            image.save(save_path+ str(num) + '.png')

        controlnet_outs, sds_w = process(
            controlnet,
            ddim_sampler,
            image_np,
            prompt = args.prompt,
            a_prompt = 'best quality,sharp',
            n_prompt = 'blur, lowres, bad anatomy, bad hands, cropped, worst quality',
            num_samples = 1,
            image_resolution = min(image_np.shape[0], image_np.shape[1]),
            ddim_steps = 50,
            guess_mode = False,
            strength = 0.8,
            scale = 1.0,
            eta = 1.0,
            denoise_strength = 0.2
        )

        best_controlnet_out = controlnet_outs[0]
        image = transform(Image.fromarray(best_controlnet_out))
        image2 = ToPILImage()(image)
        image2.save(save_path_diff + str(num) + '.png')
        diff_outputs.append(image)


    # # #--------------------------------cache-----------------------------------------#              
    # diff_images = sorted(os.listdir(dataset.model_path + '/diff/' + str(first_iter) + '_gs/'), key=extract_number)
    # for diff_image_name in diff_images:
    #     diff_image = Image.open(os.path.join(dataset.model_path + '/diff/' + str(first_iter) + '_diff/', diff_image_name))
    #     diff_gt_image = transform(diff_image)
    #     diff_outputs.append(diff_gt_image)
    # # #--------------------------------cache-----------------------------------------#   


    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)


    gaussians.update_learning_rate(100)
    viewpoint_stack = None
    ema_loss_for_log, ema_depth_loss_for_log, ema_mask_loss_for_log, ema_normal_loss_for_log = 0.0, 0.0, 0.0, 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):  
        if iteration > checkpoint_iterations[-1]:
            break
        if iteration in iter_rounds:
                print("another round of diffuse!")
                diff_outputs = []
                save2_path = dataset.model_path + '/diff/' + str(iteration) + '_gs/'
                save2_path_diff = dataset.model_path + '/diff/' + str(iteration) + '_diff/'
                if not os.path.exists(save2_path):
                    os.makedirs(save2_path)
                if not os.path.exists(save2_path_diff):
                    os.makedirs(save2_path_diff)
                for num in range(len(virtual_cameras_R)):
                    camera_center = virtual_camera_center[num]
                    world_view_transform = virtual_world_view_transform[num]
                    full_proj_transform = virtual_full_proj_transform[num]
                    render_pkg_point = render_point(standard_cam, gaussians, \
                                camera_center, world_view_transform, full_proj_transform, \
                                pipe, background) 
                    rendered_image = render_pkg_point["render"]
                    if rendered_image.shape[2] >= 1600:
                        rendered_image = trans(rendered_image)
                    image_np = rendered_image.detach().cpu().numpy() * 255
                    image_np = np.uint8(np.transpose(np.clip(image_np, 0, 255), (1, 2, 0)))
                    if visualize:
                        image = Image.fromarray(image_np)
                        image.save(save2_path+ str(num) + '.png')

                    controlnet_outs, sds_w = process(
                        controlnet,
                        ddim_sampler,
                        image_np,
                        prompt = args.prompt,
                        a_prompt = 'best quality,sharp',
                        n_prompt = 'blur, lowres, bad anatomy, bad hands, cropped, worst quality',
                        num_samples = 1,
                        image_resolution = min(image_np.shape[0], image_np.shape[1]),
                        ddim_steps = 50,
                        guess_mode = False,
                        strength = 0.8,
                        scale = 1.0,
                        eta = 1.0,
                        denoise_strength = 0.2
                    )
                    best_controlnet_out = controlnet_outs[0]
                    image = transform(Image.fromarray(best_controlnet_out))
                    image2 = ToPILImage()(image)
                    image2.save(save2_path_diff + str(num) + '.png')
                    diff_outputs.append(image)

                gaussians.update_learning_rate(100)

        iter_start.record()
        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam: Camera = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        rendered_image: torch.Tensor
        rendered_image, viewspace_point_tensor, visibility_filter, radii = (
                                                                    render_pkg["render"], 
                                                                    render_pkg["viewspace_points"], 
                                                                    render_pkg["visibility_filter"], 
                                                                    render_pkg["radii"])
        
        rendered_mask: torch.Tensor = render_pkg["mask"]
        rendered_depth: torch.Tensor = render_pkg["depth"]
        rendered_middepth: torch.Tensor = render_pkg["middepth"]
        rendered_normal: torch.Tensor = render_pkg["normal"]
        depth_distortion: torch.Tensor = render_pkg["depth_distortion"]
        
        gt_image = viewpoint_cam.original_image
        edge = viewpoint_cam.edge
        gt_depth = viewpoint_cam.gt_depth.unsqueeze(0).cuda() if opt.opt_train_depth else None
        gt_normal = viewpoint_cam.gt_normal.cuda() if opt.opt_train_normal else None
        
        if dataset.use_decoupled_appearance:
            Ll1_render = L1_loss_appearance(rendered_image, gt_image, gaussians, viewpoint_cam.uid)
        else:
            Ll1_render = l1_loss(rendered_image, gt_image)

        if iteration % 1000 == 0 or iteration == first_iter:
            with torch.no_grad():
                render_pkg2 = render(viewpoint_cam, gaussians, pipe, background)
                rendered_img = render_pkg2["render"]
                rendered_depth_gradient = depth_to_normal(viewpoint_cam, rendered_depth)[0]
                rendered_middepth_gradient = depth_to_normal(viewpoint_cam, rendered_middepth)[0]
                plt.figure()
                plt.subplot(3,2,1)
                plt.imshow(rendered_image.clamp(0,1).detach().cpu().numpy().transpose(1,2,0))
                plt.subplot(3,2,2)
                plt.imshow(rendered_img.clamp(0,1).detach().cpu().numpy().transpose(1,2,0))
                plt.subplot(3,2,3)
                plt.imshow((-rendered_depth_gradient.detach().cpu().numpy()+1)/2)
                plt.subplot(3,2,4)
                plt.imshow((-rendered_normal.detach().cpu().numpy().transpose(1,2,0)+1)/2)
                plt.subplot(3,2,5)
                plt.imshow((-rendered_middepth_gradient.detach().cpu().numpy()+1)/2)
                plt.subplot(3,2,6)
                plt.imshow(apply_depth_colormap(rendered_depth[0,...,None], rendered_mask[0,...,None]).detach().cpu().numpy())
                plt.savefig(os.path.join(dataset.model_path,"test_%d.png"%iteration), bbox_inches='tight', dpi=1000)
                plt.close()        
        
        loss = 0 
        
        random_pose = randint(0, len(virtual_cameras_R) - 1)
        diff_gt_image = diff_outputs[random_pose].cuda()
        camera_center = virtual_camera_center[random_pose]
        world_view_transform = virtual_world_view_transform[random_pose]
        full_proj_transform = virtual_full_proj_transform[random_pose]
        render_test = render_point(viewpoint_cam, gaussians, \
                    camera_center, world_view_transform, full_proj_transform, \
                    pipe, background) 
        diff_rendered_image, viewspace_point_tensor, visibility_filter, radii = (
                                                                render_test["render"], 
                                                                render_test["viewspace_points"], 
                                                                render_test["visibility_filter"], 
                                                                render_test["radii"])
        random_T = torch.from_numpy(virtual_cameras_t[random_pose]).to(train_camera.data_device)
        distances = get_dis_from_ts(random_T, torch.stack(train_T_list))
        distance_weight = min(1., 16 * distances[0].cpu().item() / max_cam_dis) #4
        # print(16 * distances[0].cpu().item() / max_cam_dis)
        L_pr = l1_loss(diff_rendered_image, diff_gt_image)
        loss += L_pr * distance_weight 

        if visualize and iteration % 100 == 0:
            vis_dir = os.path.join(tb_writer.log_dir,'vis')
            if not os.path.exists(vis_dir):
                os.makedirs(vis_dir) 
            gt_show = cv2.cvtColor((gt_image.permute(1,2,0).detach().cpu().numpy()*255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            cv2.imwrite(vis_dir + '/render_gt_image.png', gt_show)
            point_show = cv2.cvtColor( (diff_gt_image*255).permute(1,2,0).cpu().numpy().astype(np.uint8), cv2.COLOR_RGB2BGR)
            cv2.imwrite(vis_dir + '/diff_gt_image.png', point_show)
            splat_show = cv2.cvtColor((diff_rendered_image.detach().permute(1,2,0).cpu().numpy()*255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            cv2.imwrite(vis_dir + '/diff_render_image.png', splat_show)
        if tb_writer is not None:
                tb_writer.add_scalar('loss/diff_loss', L_pr * distance_weight , iteration)

        if opt.opt_train_depth:
            pearson_loss = pearson_depth_loss(rendered_depth[0], gt_depth[0])
            lp_loss = local_pearson_loss(rendered_depth[0], gt_depth[0], 128, 0.5)
            depth_loss = (pearson_loss + lp_loss) * 0.1
            loss += depth_loss
            if tb_writer is not None:
                tb_writer.add_scalar('loss/depth_loss', pearson_loss, iteration)
        
        if opt.opt_train_normal:
            if viewpoint_cam.gt_normal is not None:
                normal_gt = gt_normal.permute(2, 0, 1)
                normal_gt = vis_surface_normal_inverse(normal_gt)
                filter_mask = (normal_gt != -10)[0, :, :].to(torch.bool)
                l1_normal = torch.abs(rendered_normal - normal_gt).sum(dim=0)[filter_mask].mean()
                cos_normal = (1. - torch.sum(rendered_normal * normal_gt, dim = 0))[filter_mask].mean()
                normal_loss = 0.01 * l1_normal + 0.01 * cos_normal
                loss += normal_loss 
                if tb_writer is not None:
                    tb_writer.add_scalar('loss/normal_loss', normal_loss, iteration)

        if visualize and opt.opt_train_depth and opt.opt_train_normal and iteration % 100 == 0:
            # gs-rendered depth map
            depth_np = rendered_depth.detach().cpu().numpy().squeeze(0)
            normalized_depth =(depth_np - depth_np.min()) / (depth_np.max() - depth_np.min())   
            colored_depth = cv2.cvtColor(normalized_depth*255, cv2.COLOR_GRAY2BGR).astype(np.uint8)
            colored_depth = cv2.applyColorMap(colored_depth, cv2.COLORMAP_JET)
            cv2.imwrite(vis_dir + '/gs-render-depth.png', colored_depth)
            gt_depth = viewpoint_cam.gt_depth.detach().cpu().numpy()
            # mono depth map
            normalized_depth_gt = (gt_depth - gt_depth.min()) / (gt_depth.max() - gt_depth.min())    
            colored_depth_gt = cv2.cvtColor(normalized_depth_gt*255, cv2.COLOR_GRAY2BGR).astype(np.uint8)
            colored_depth_gt = cv2.applyColorMap(colored_depth_gt, cv2.COLORMAP_JET)
            cv2.imwrite(vis_dir + '/mono_depth.png', colored_depth_gt)
            # gs-rendered normal & mono normal
            re_normal = rendered_normal.permute(1,2,0)
            normal_np = vis_surface_normal(re_normal.detach())
            cv2.imwrite(vis_dir + '/gs-render-normal.png', normal_np*255)
            cv2.imwrite(vis_dir + '/mono_normal.png', gt_normal.detach().cpu().numpy())

        if iteration >= opt.regularization_from_iter:
            # depth distortion loss
            lambda_distortion = opt.lambda_distortion
            depth_distortion = torch.where(rendered_mask>0,depth_distortion/(rendered_mask * rendered_mask).detach(),0)
            distortion_map = depth_distortion[0] * edge
            distortion_loss = distortion_map.mean()

            # normal consistency loss
            rendered_depth = rendered_depth / rendered_mask
            rendered_depth = torch.nan_to_num(rendered_depth, 0, 0)
            depth_middepth_normal, _ = depth_double_to_normal(viewpoint_cam, rendered_depth, rendered_middepth)
            depth_ratio = 0.6
            rendered_normal = torch.nn.functional.normalize(rendered_normal, p=2, dim=0)
            rendered_normal = rendered_normal.permute(1,2,0)
            normal_error_map = (1 - (rendered_normal.unsqueeze(0) * depth_middepth_normal).sum(dim=-1))
            depth_normal_loss = (1-depth_ratio) * normal_error_map[0].mean() + depth_ratio * normal_error_map[1].mean()
            lambda_depth_normal = opt.lambda_depth_normal
        else:
            lambda_distortion = 0
            lambda_depth_normal = 0
            distortion_loss = torch.tensor(0,dtype=torch.float32,device="cuda")
            depth_normal_loss = torch.tensor(0,dtype=torch.float32,device="cuda")
            
        rgb_loss = (1.0 - opt.lambda_dssim) * Ll1_render + opt.lambda_dssim * (1.0 - ssim(rendered_image, gt_image.unsqueeze(0)))
        
        loss += (rgb_loss + depth_normal_loss * lambda_depth_normal + distortion_loss * lambda_distortion)
        loss.backward()

        iter_end.record()


        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_depth_loss_for_log = 0.4 * distortion_loss.item() + 0.6 * ema_depth_loss_for_log
            ema_normal_loss_for_log = 0.4 * depth_normal_loss.item() + 0.6 * ema_normal_loss_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{4}f}", "loss_dep": f"{ema_depth_loss_for_log:.{4}f}", "loss_normal": f"{ema_normal_loss_for_log:.{4}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            if iteration in testing_iterations:# save the render results for repairing
                  save_path = dataset.model_path + '/diff/' + str(iteration) + '_gs/'
                  if not os.path.exists(save_path):
                        os.makedirs(save_path)
                  for random_pose in range(len(virtual_cameras_R)):
                        diff_gt_image = diff_outputs[random_pose].cuda()
                        camera_center = virtual_camera_center[random_pose]
                        world_view_transform = virtual_world_view_transform[random_pose]
                        full_proj_transform = virtual_full_proj_transform[random_pose]
                        render_pkg = render_point(viewpoint_cam, gaussians, \
                                    camera_center, world_view_transform, full_proj_transform, \
                                    pipe, background) 
                        rendered_image = render_pkg["render"]
                        image_np = rendered_image.detach().cpu().numpy() * 255
                        image_np = np.uint8(np.transpose(np.clip(image_np, 0, 255), (1, 2, 0)))
                        image = Image.fromarray(image_np)
                        image.save(save_path + str(random_pose) + '.png')


            # Log and save
            training_report(tb_writer, iteration, Ll1_render, loss, distortion_loss, depth_normal_loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.05, scene.cameras_extent, size_threshold)
                    gaussians.compute_3D_filter(cameras=trainCameras)

                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()
                
            if iteration % 100 == 0 and iteration > opt.densify_until_iter:
                if iteration < opt.iterations - 100:
                    # don't update in the end of training
                    gaussians.compute_3D_filter(cameras=trainCameras)

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, depth_loss, normal_loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/depth_loss', depth_loss.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/normal_loss', normal_loss.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                ssim_test = 0.0
                lpips_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    if config['name'] == 'test' and iteration in testing_iterations:
                        continue
                    else:
                        render_result = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                        image = torch.clamp(render_result["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image, 0.0, 1.0)
                    if tb_writer and idx % 4 == 0:
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/normal".format(viewpoint.image_name), render_result["normal"][None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/depth".format(viewpoint.image_name), render_result["depth"][None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    print(psnr(image, gt_image).mean().double())
                    ssim_test += ssim(image, gt_image).mean().double()
                    lpips_test += lpips(image, gt_image).mean().double()
 
                psnr_test /= len(config['cameras'])
                ssim_test /= len(config['cameras'])
                lpips_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
   
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {} SSIM {} LPIPS {}".format(iteration, config['name'], l1_test, psnr_test, ssim_test, lpips_test))
                if config["name"] == "test":
                    with open(scene.model_path + "/chkpnt" + str(iteration) + ".txt", "w") as file_object:
                        print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test), file=file_object)
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - ssim', ssim_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - lpips', lpips_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams360(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[12000, 18000, 24000, 30000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[12000, 18000, 24000, 30000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[12000, 18000, 24000, 30000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--iterative_round", type=str, default = 3)
    # diffusion parsers
    parser.add_argument('--model_name', type=str, default='control_v11f1e_sd15_tile')
    parser.add_argument('--image_size', type=int, default=512)
    parser.add_argument('--prompt', type=str, default='high quality, sharp outside scene, a statue of a family in the centre')
    parser.add_argument('--exp_name', type=str, default=f'outputs/controlnet_finetune/family')
    parser.add_argument('--bg_white', action='store_true', default=False)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--only_mid_control', action='store_true', default=False)
    parser.add_argument('--train_lora', default=True)
    parser.add_argument('--lora_rank', type=int, default=64)
    parser.add_argument('--use_prompt_list', action='store_true', default=False)
    parser.add_argument('--manual_noise_reduce_start', type=int, default=100)
    parser.add_argument('--manual_noise_reduce_gamma', type=float, default=0.995)

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
     
    #load diffusion repair model
    model = create_model(f'./models/{args.model_name}.yaml').cpu()
    model.load_state_dict(load_state_dict('./models/v1-5-pruned.ckpt', location='cpu'), strict=False)
    model.load_state_dict(load_state_dict(f'./models/{args.model_name}.pth', location='cpu'), strict=False)
    model.learning_rate = args.learning_rate
    model.sd_locked = True
    model.only_mid_control = args.only_mid_control
    model.train_lora = True

    lora_config = {
        nn.Embedding: {
            "weight": partial(LoRAParametrization.from_embedding, rank=args.lora_rank)
        },
        nn.Linear: {
            "weight": partial(LoRAParametrization.from_linear, rank=args.lora_rank)
        },
        nn.Conv2d: {
            "weight": partial(LoRAParametrization.from_conv2d, rank=args.lora_rank)
        }
    }

    for name, module in model.model.diffusion_model.named_modules():
        if name.endswith('transformer_blocks'):
            add_lora(module, lora_config=lora_config)

    for name, module in model.control_model.named_modules():
        if name.endswith('transformer_blocks'):
            add_lora(module, lora_config=lora_config)

    add_lora(model.cond_stage_model, lora_config=lora_config)

    model.load_state_dict(load_state_dict(args.exp_name + '/ckpts-lora/lora-step=1799.ckpt', location='cuda'), strict=False)
    controlnet = model.cuda()
    ddim_sampler = DDIMSampler(controlnet)

    print("Optimizing " + args.model_path)
    safe_state(args.quiet)

    training(dataset=lp.extract(args), 
             opt=op.extract(args), 
             pipe=pp.extract(args), 
             testing_iterations=args.test_iterations, 
             saving_iterations=args.save_iterations, 
             checkpoint_iterations=args.checkpoint_iterations, 
             checkpoint=args.start_checkpoint, 
             debug_from=args.debug_from,
             controlnet=controlnet,
             ddim_sampler=ddim_sampler,
             iter_num = args.iterative_round)

    # All done
    print("\nTraining complete.")
