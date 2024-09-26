# for feed-forward scene(such as 3 input images), this script optimize smaller iterations

import os
import torch
# import torch.nn.functional as F
from random import randint
from utils.loss_utils import l1_loss, ssim, lpr_loss, pearson_depth_loss, local_pearson_loss
from utils.graphics_utils import vis_surface_normal_inverse, vis_surface_normal, load_pointcloud, interpolate_camera_poses
from gaussian_renderer import render, render_point
import sys
import cv2
from lpipsPyTorch import lpips
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
import torchvision
from tqdm import tqdm
import numpy as np
from utils.image_utils import psnr
from utils.graphics_utils import depth_double_to_normal, extract_number
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParamsinstant
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
from PIL import Image
from glob import glob
from scene.cameras import Camera
from matplotlib import pyplot as plt
from utils.depth_utils import depth_to_normal
from utils.vis_utils import apply_depth_colormap
from pytorch3d.renderer import (
    PerspectiveCameras,
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    NormWeightedCompositor
)
import random
import torchvision.transforms as transforms


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

def training(dataset, save_dir, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    set_seed(10)
    first_iter = 0
    dataset.model_path = save_dir

    tb_writer = prepare_output_and_logger(dataset)
    visualize = True 
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, opt.opt_train_depth, opt.opt_train_normal, gap = pipe.interval)
    C, H, W = scene.train_cameras[1][1].original_image.shape
    # use virtual cameras to regularize novel views, 
    # here a modified pytorch3d is used to render point maps, 
    # Marigold is used to render depth maps

    if visualize:
        point_dir = os.path.join(tb_writer.log_dir,'pointrender')
        if not os.path.exists(point_dir):
            os.makedirs(point_dir)    
    num_virtual_poses = 10    
    point_render_maps = []
    point_cloud = load_pointcloud(dataset.source_path, device=dataset.data_device)
    virtual_cameras_R,  virtual_cameras_t, virtual_camera_center, virtual_world_view_transform, virtual_full_proj_transform \
        = interpolate_camera_poses(scene.train_cameras[1.0], num_virtual_poses)
    
    focal = scene.train_cameras[1.0][0].intrinsic[0][0].item()
    H1, W1 = int(scene.train_cameras[1.0][0].intrinsic[1][2].item() * 2), int(scene.train_cameras[1.0][0].intrinsic[0][2].item() * 2)
    raster_settings = PointsRasterizationSettings(
        image_size=[H1,W1], 
        radius = 0.01,
        points_per_pixel = 20
    )

    for num in range(len(virtual_cameras_R)):
        R=virtual_cameras_R[num].astype(np.float32)
        T=virtual_cameras_t[num].astype(np.float32)
        camera = PerspectiveCameras(focal_length=torch.tensor([[focal, focal]]),
                principal_point=torch.tensor([[W1/2, H1/2]]),
                R=torch.tensor(R).unsqueeze(0),
                T=torch.tensor(T).unsqueeze(0),
                device=dataset.data_device,
                in_ndc=False,
                image_size = [(H1,W1)]
                )
        # get point render maps
        renderer = PointsRenderer(
            rasterizer=PointsRasterizer(cameras=camera, raster_settings=raster_settings),
            compositor=NormWeightedCompositor(background_color=(1,1,1))
        )
        images = renderer(point_cloud)
        image = images[0, ..., :3].cpu().numpy()
        image_normalized = np.fliplr(np.flipud(image / np.max(image)))
        point_render_maps.append(image_normalized.copy())
        image = Image.fromarray((image_normalized * 255).astype(np.uint8))
        if visualize:
            image.save(point_dir + '/image' + str(num) + ".png")

        
        
        # -------------------when use cache imgs , use this code---------------------------------
        point_render_maps = []
        EXTENSION_LIST = [".jpg", ".jpeg", ".png"]
        point_rgb_path = point_dir 

        rgb_filename_list = glob(os.path.join(point_rgb_path, "*"))
        rgb_filename_list = [
            f for f in rgb_filename_list if os.path.splitext(f)[1].lower() in EXTENSION_LIST
        ]
        rgb_filename_list = sorted(rgb_filename_list, key=lambda x: int(x.split('image')[1].split('.png')[0]))
        if W >= 1600:
            trans = transforms.Resize([H, W], antialias=True)
        with torch.no_grad():
            for rgb_path in rgb_filename_list:
                input_image = Image.open(rgb_path)
                to_tensor = transforms.ToTensor()

                input_image = to_tensor(input_image)
                input_image = input_image
                if W >= 1600:
                    input_image = trans(input_image)
                point_render_maps.append(input_image.permute(1,2,0).to(dataset.data_device) )
        #-------------------when use cache imgs , use this code---------------------------------
    

    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    trainCameras = scene.getTrainCameras().copy()
    gaussians.compute_3D_filter(cameras=trainCameras)

    viewpoint_stack = None
    ema_loss_for_log, ema_depth_loss_for_log, ema_mask_loss_for_log, ema_normal_loss_for_log = 0.0, 0.0, 0.0, 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    # set sample dir
    viewpoint_stack = scene.getTrainCameras().copy()
    for i in range(len(viewpoint_stack)):
        sample_save_path = os.path.join(tb_writer.log_dir, str(i))
        os.makedirs(sample_save_path, exist_ok=True)

    for iteration in range(first_iter, opt.iterations + 1):        
        if iteration > checkpoint_iterations[-1]:
            break
        iter_start.record()
        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        randindex = randint(0, len(viewpoint_stack)-1)
        viewpoint_cam: Camera = viewpoint_stack.pop(randindex)
        save_dir = os.path.join(tb_writer.log_dir, str(viewpoint_cam.uid))
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
        gt_depth = viewpoint_cam.gt_depth.unsqueeze(0).to(dataset.data_device) if opt.opt_train_depth else None
        gt_normal = viewpoint_cam.gt_normal.to(dataset.data_device)  if opt.opt_train_normal else None


        
        if dataset.use_decoupled_appearance:
            Ll1_render = L1_loss_appearance(rendered_image, gt_image, gaussians, viewpoint_cam.uid)
        else:
            Ll1_render = l1_loss(rendered_image, gt_image)

        
        # cache the rendered images in the rendering progress for diffusion lora use.
        if iteration < 2000:
            torchvision.utils.save_image(rendered_image, os.path.join(save_dir, f'sample_{iteration}.png'))

        loss = 0 
       
        random_pose = randint(0, len(point_render_maps)  - 1)
        camera_center = virtual_camera_center[random_pose]
        world_view_transform = virtual_world_view_transform[random_pose]
        full_proj_transform = virtual_full_proj_transform[random_pose]
        render_pkg_point = render_point(viewpoint_cam, gaussians, \
                    camera_center, world_view_transform, full_proj_transform, \
                    pipe, background) 
        image = render_pkg_point["render"].permute(1,2,0)

        point_image = point_render_maps[random_pose]
        L_pr = lpr_loss(image, point_image, device=dataset.data_device)
        loss += L_pr * 0.3
        if tb_writer is not None:
                tb_writer.add_scalar('loss/point_render_loss', L_pr, iteration)
                    

        if visualize and iteration % 100 == 0:
            vis_dir = os.path.join(tb_writer.log_dir,'vis')
            if not os.path.exists(vis_dir):
                os.makedirs(vis_dir)  
            gt_show = cv2.cvtColor((gt_image.permute(1,2,0).detach().cpu().numpy()*255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            cv2.imwrite(vis_dir + '/render_gt_image.png', gt_show)
            gt_show = cv2.cvtColor((rendered_image.permute(1,2,0).detach().cpu().numpy()*255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            cv2.imwrite(vis_dir + '/render_gs_image.png', gt_show)
            if isinstance(point_image, torch.Tensor):
                point_image_vis = point_image.cpu().numpy()
                point_show = cv2.cvtColor( (point_image_vis*255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            else:
                point_show = cv2.cvtColor( (point_image*255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            cv2.imwrite(vis_dir + '/render_point_image.png', point_show)
            splat_show = cv2.cvtColor((image.detach().cpu().numpy()*255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            cv2.imwrite(vis_dir + '/render_splat_image.png', splat_show)
        if tb_writer is not None:
                tb_writer.add_scalar('loss/point_render_loss', L_pr, iteration)
                 
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
            lambda_depth_normal = 0
            depth_normal_loss = torch.tensor(0,dtype=torch.float32,device="cuda")
            
        rgb_loss = (1.0 - opt.lambda_dssim) * Ll1_render + opt.lambda_dssim * (1.0 - ssim(rendered_image, gt_image.unsqueeze(0)))
        
        loss += (rgb_loss + depth_normal_loss * lambda_depth_normal)
        loss.backward()
 
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_depth_loss_for_log = 0
            ema_normal_loss_for_log = 0.4 * depth_normal_loss.item() + 0.6 * ema_normal_loss_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{4}f}", "loss_dep": f"{ema_depth_loss_for_log:.{4}f}", "loss_normal": f"{ema_normal_loss_for_log:.{4}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()
            
            # Log and save
            training_report(tb_writer, iteration, Ll1_render, loss, 0, depth_normal_loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
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
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.02, scene.cameras_extent, size_threshold)#0.05
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
                    render_result = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(render_result["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image, 0.0, 1.0)
                    if tb_writer and idx % 10 == 0:
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/normal".format(viewpoint.image_name), render_result["normal"][None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/depth".format(viewpoint.image_name), render_result["depth"][None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
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
    op = OptimizationParamsinstant(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[1000, 2000, 3000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[1000, 2000, 3000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[1000, 2000, 3000])
    parser.add_argument("--start_checkpoint", type=str, default =None)
    parser.add_argument("--save",type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    # torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(dataset=lp.extract(args), 
             save_dir = args.save,
             opt=op.extract(args), 
             pipe=pp.extract(args), 
             testing_iterations=args.test_iterations, 
             saving_iterations=args.save_iterations, 
             checkpoint_iterations=args.checkpoint_iterations, 
             checkpoint=args.start_checkpoint, 
             debug_from=args.debug_from)

    # All done
    print("\nTraining complete.")
