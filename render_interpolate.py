import os
import torch
import torch.nn.functional as F
from random import randint

from utils.graphics_utils import interpolate_camera_poses
from gaussian_renderer import render, render_point 
import sys
import imageio
from lpipsPyTorch import lpips
from scene import Scene, GaussianModel
from utils.general_utils import safe_state

from tqdm import tqdm
import numpy as np
from argparse import ArgumentParser
from glob import glob
from scene.cameras import Camera
from matplotlib import pyplot as plt
from arguments import ModelParams, PipelineParams, OptimizationParams360
import torchvision


def images_to_video(image_folder, output_video_path, fps=30):
    """
    Convert images in a folder to a video.

    Args:
    - image_folder (str): The path to the folder containing the images.
    - output_video_path (str): The path where the output video will be saved.
    - fps (int): Frames per second for the output video.
    """
    images = []

    for filename in sorted(os.listdir(image_folder)):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.JPG', '.PNG')):
            image_path = os.path.join(image_folder, filename)
            image = imageio.imread(image_path)
            images.append(image)

    imageio.mimwrite(output_video_path, images, fps=fps)




def render_sets(dataset, save_dir, opt, pipe, checkpoint_iterations, checkpoint):
    dataset.model_path = save_dir
    opt_train_depth = True
    opt_train_normal = True
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, opt_train_depth, opt_train_normal, load_iteration = checkpoint_iterations, gap = pipe.interval)
    gaussians.training_setup(opt)
    assert(checkpoint != None)    
    (model_params, first_iter) = torch.load(checkpoint)
    gaussians.restore(model_params, opt)
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")    

    trainCameras = scene.getTrainCameras().copy()
    gaussians.compute_3D_filter(cameras=trainCameras)      

    render_dir = os.path.join(dataset.model_path, f'interpolate/ours_{checkpoint_iterations}/renders')
    if not os.path.exists(render_dir):
            os.makedirs(render_dir)    
    num_virtual_poses = 30
    virtual_cameras_R,  virtual_cameras_t, virtual_camera_center, virtual_world_view_transform, virtual_full_proj_transform \
                  = interpolate_camera_poses(scene.train_cameras[1.0], num_virtual_poses)
    
    viewpoint_stack = scene.getTrainCameras().copy()
    randindex = randint(0, len(viewpoint_stack)-1)
    viewpoint_cam: Camera = viewpoint_stack.pop(randindex)

    for num in range(len(virtual_cameras_R)):
            camera_center = virtual_camera_center[num]
            world_view_transform = virtual_world_view_transform[num]
            full_proj_transform = virtual_full_proj_transform[num]
            render_pkg_point = render_point(viewpoint_cam, gaussians, \
                        camera_center, world_view_transform, full_proj_transform, \
                        pipe, background) 
            image = render_pkg_point["render"].permute(1,2,0)
            torchvision.utils.save_image(image.permute(2, 0, 1), os.path.join(render_dir, "{0:05d}".format(num) + ".png") )


    if args.get_video:
        image_folder = os.path.join(dataset.model_path, f'interpolate/ours_{checkpoint_iterations}/renders')
        output_video_file = os.path.join(dataset.model_path, f'{checkpoint_iterations}_render_video.mp4')
        images_to_video(image_folder, output_video_file, fps=30)


if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams360(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--get_video", action="store_true", default=True)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--start_checkpoint", type=str, default = None)

    args = parser.parse_args(sys.argv[1:])

    print("Rendering " + args.model_path)
    save_dir = os.path.dirname(args.start_checkpoint)
    checkpoint_iterations = args.start_checkpoint.split("chkpnt")[1].split(".")[0]
    render_sets(
        dataset=lp.extract(args), 
        save_dir = save_dir,
        opt=op.extract(args), 
        pipe=pp.extract(args), 
        checkpoint_iterations=checkpoint_iterations, 
        checkpoint=args.start_checkpoint, 
    )

