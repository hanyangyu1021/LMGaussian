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
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
import torch
import torch.nn as nn

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, opt_depth=False, opt_normal=False, \
                 load_iteration=None, shuffle=False, resolution_scales=[1.0], gap = 1):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, True)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval, gap = gap)
        elif os.path.exists(os.path.join(args.source_path, "llff.txt")):
            print("Found llff.txt file, assuming LLFF data set!")
            scene_info = sceneLoadTypeCallbacks["LLFF"](args.source_path, args.white_background, opt_depth, opt_normal, args.eval)
        else:
            print("Found pair.txt file, assuming DUST3R data set!")
            scene_info = sceneLoadTypeCallbacks["DUST3R"](args.source_path, args.white_background, opt_depth, opt_normal, args.eval)

        
        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)


        # train poses
        num_train_samples = len(scene_info.train_cameras)
        self.name_list_according_to_cameras = []
        for i in range(num_train_samples):
            image_name = scene_info.train_cameras[i].image_name
            self.name_list_according_to_cameras.append(image_name)
        # test poses
        num_test_samples = len(scene_info.test_cameras)
        self.name_list_according_to_cameras_test = []
        for i in range(num_test_samples):
            image_name = scene_info.test_cameras[i].image_name
            self.name_list_according_to_cameras_test.append(image_name)


        self.R_q = nn.Parameter(torch.zeros(size=(num_train_samples, 3), dtype=torch.float32), requires_grad = True)
        self.T_q = nn.Parameter(torch.zeros(size=(num_train_samples, 3), dtype=torch.float32), requires_grad = True)
        parameters = [{'params': [self.R_q], 'lr': 5e-4, "name": "R_q"}, {'params': [self.T_q], 'lr': 5e-4, "name": "T_q"}]
        self.optimizer_pose = torch.optim.Adam(parameters, lr=5e-4)
        self.scheduler_pose = torch.optim.lr_scheduler.MultiStepLR(self.optimizer_pose, 
                                                                milestones=list(range(0, 800, 30)),
                                                                gamma=0.9, last_epoch=-1)
        self.R_q_test = nn.Parameter(torch.zeros(size=(num_test_samples, 3), dtype=torch.float32), requires_grad = True)
        self.T_q_test = nn.Parameter(torch.zeros(size=(num_test_samples, 3), dtype=torch.float32), requires_grad = True)
        parameters_test = [{'params': [self.R_q_test], 'lr': 5e-4, "name": "R_q_test"}, {'params': [self.T_q_test], 'lr': 5e-4, "name": "T_q_test"}]
        self.optimizer_test = torch.optim.Adam(parameters_test, lr=5e-3)
        self.scheduler_test = torch.optim.lr_scheduler.MultiStepLR(self.optimizer_test, 
                                                                milestones=list(range(0 , 200, 20)),
                                                                gamma=0.9, last_epoch=-1)      



        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print(f"Loading Training Cameras: {len(self.train_cameras[resolution_scale])} .")
            
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)
            print(f"Loading Test Cameras: {len(self.test_cameras[resolution_scale])} .")

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)
            self.gaussians.init_RT_seq(self.train_cameras)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))


    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
    
    def step(self):
        self.optimizer_pose.step()
        self.optimizer_pose.zero_grad()
        
    def step_test(self):
        self.optimizer_test.step()
        self.optimizer_test.zero_grad()   