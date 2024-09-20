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
import sys
from PIL import Image
from typing import NamedTuple, Optional
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
from scipy.spatial.transform import Rotation
import cv2
import trimesh
import open3d as o3d
import re

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    depth: np.array
    mask: np.array = None
    intrinsics: np.array = None
    extrinsics: np.array = None
    normal: Optional[np.ndarray] = None

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str


def fetchPly_scale(path, scale):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T * scale
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def read_pfm(filename: str):
    """Read a depth map from a .pfm file

    Args:
        filename: .pfm file path string

    Returns:
        data: array of shape (H, W, C) representing loaded depth map
        scale: float to recover actual depth map pixel values
    """
    file = open(filename, "rb")  # treat as binary and read-only

    header = file.readline().decode("utf-8").rstrip()
    if header == "PF":
        color = True
    elif header == "Pf": # depth is Pf
        color = False
    else:
        raise Exception("Not a PFM file.")

    dim_match = re.match(r"^(\d+)\s(\d+)\s$", file.readline().decode("utf-8"))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception("Malformed PFM header.")

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = "<"
        scale = -scale
    else:
        endian = ">"  # big-endian

    data = np.fromfile(file, endian + "f")
    shape = (height, width, 3) if color else (height, width, 1)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    file.close()
    return data, scale


def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)
        
        # # read depth image
        # depth_path = os.path.join(os.path.dirname(images_folder)+'/depths', image_name + ".png")
        # # for Replica data, the depth scale is 4000
        # depth = np.array(Image.open(depth_path)) / 4000.0
        # read depth image
        depth_path = os.path.join(os.path.dirname(images_folder)+'/depth_est', "00"+image_name + ".pfm")
        confidence_path = os.path.join(os.path.dirname(images_folder)+'/confidence', "00"+image_name + ".pfm")
        if os.path.exists(depth_path):
            depth, _ = read_pfm(depth_path)
            depth = depth[...,0]
            depth = Image.fromarray(depth)
            confidence, _ = read_pfm(confidence_path)
            mask = (confidence>0.9).astype(np.uint8)*255
            image = Image.fromarray(np.concatenate([np.asarray(image),mask],axis=-1), "RGBA")
        else:
            depth = None
        # depth_mask_path = os.path.join(os.path.dirname(images_folder)+'/mask', "00"+image_name + "_final.png")
        # if os.path.exists(depth_mask_path):
        #     mask = cv2.imread(depth_mask_path,-1)
        #     image = Image.fromarray(np.concatenate([np.asarray(image),mask[...,None]],axis=-1), "RGBA")
        # for Replica data, the depth scale is 4000
        intrinsic = np.array([[focal_length_x,0,width/2],[0,focal_length_y,height/2],[0,0,1]])
        extrinsic = np.eye(4)
        extrinsic[:3,:3] = qvec2rotmat(extr.qvec)
        extrinsic[:3,3] = T

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image, depth=depth, 
                              image_path=image_path, image_name=image_name, width=width, height=height, intrinsics=intrinsic, extrinsics=extrinsic)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def fetchOpen3DPly(path):
    plydata = o3d.io.read_point_cloud(path)
    positions = np.asarray(plydata.points)
    colors = np.asarray(plydata.colors)
    if plydata.has_normals():
        normals = np.asarray(plydata.normals)
    else:
        normals = np.zeros_like(positions)
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, eval, llffhold=13):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    # if eval:
    #     train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    #     test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
    # else:
    train_cam_infos = cam_infos
    test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)
    print(f'cameras extent: {nerf_normalization["radius"]}')

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        # pcd = None
        try:
            pcd = trimesh.load(ply_path)
            point_id = np.random.choice(np.arange(len(pcd.vertices)), 1200000)
            pcd = BasicPointCloud(points=pcd.vertices[point_id], colors=pcd.colors[point_id][:,:3].astype(np.float32)/255, normals=None)
        except:
            pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx


            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1], depth=None, mask=None))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png", gap = 1):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    train_cam_infos = train_cam_infos[::gap]
    print("train num:", len(train_cam_infos))
    # if not eval:
    #     train_cam_infos.extend(test_cam_infos)
    #     test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           point_cloud=pcd)
    return scene_info

def readCamerasFromDUST3R(img_path, cam_path, white_background, extension=".jpg", scale = 50, dep_path = None, nor_path = None):
    cam_infos = []
    image_files = sorted(os.listdir(img_path))
    cam_files = sorted(os.listdir(cam_path))
    if dep_path != None:
        dep_files = sorted(os.listdir(dep_path))
    if nor_path != None:
        nor_files = sorted(os.listdir(nor_path))
    idx = 0
    for image_file in image_files:
        image_name = os.path.splitext(image_file)[0]
        idx += 1
        cam_file = f"{image_name}_cam.txt"
        dep_file = f"depth_{image_name}.png"
        nor_file = f"normal_{image_name}.png"
        if cam_file in cam_files:
            image_path = os.path.join(img_path, image_file)
            camera_path = os.path.join(cam_path, cam_file)
        else :
            raise Exception("Error message: no cam file exits matched with{image_file}")
        
        image = Image.open(image_path)
        W1, H1 = image.size
        # W2, H2 = 512, 288  #320
        im_data = np.array(image.convert("RGBA"))
        bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

        norm_data = im_data / 255.0
        arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
        image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

        if  dep_path != None and dep_file in dep_files:
            depth_path = os.path.join(dep_path, dep_file)
            depth = cv2.imread(depth_path, -1)
            depth = (depth / 1000).astype(np.float32) 
            depth = cv2.resize(depth, (W1, H1), interpolation=cv2.INTER_LINEAR)
        else: 
            depth = None

        if  nor_path != None and nor_file in nor_files:
            normal_path = os.path.join(nor_path, nor_file)
            normal = cv2.imread(normal_path, -1)
            normal = normal.astype(np.float32) 
            normal = cv2.resize(normal, (W1, H1), interpolation=cv2.INTER_LINEAR)
        else: 
            normal = None

        with open(camera_path, 'r') as file:
            lines = file.readlines()

        c2w = []
        for i in range(1,5):
            line = lines[i].strip().split()
            row = [float(val) for val in line]
            c2w.append(row)
        c2w = np.array(c2w)
        w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3,:3])  
        T = w2c[:3, 3] * scale

        K = []
        for i in range(7, 10):
            line = lines[i].strip().split()
            row = [float(val) for val in line]
            K.append(row)
        K = np.array(K)
        W2, H2 = K[0][2]*2, K[1][2]*2  
        FovX = focal2fov(K[0,0],W2)
        FovY = focal2fov(K[1,1],H2)
        intrinsic = np.array([fov2focal(FovX, W1), fov2focal(FovY, H1), W1/2, H1/2])
        intrinsic = np.array([[fov2focal(FovX, W1), 0, W1/2],[0, fov2focal(FovY, H1), H1/2],[0,0,1]])
        extrinsic = np.eye(4)
        extrinsic[:3,:3] = w2c[:3,:3]
        extrinsic[:3,3] = T
        
        cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image, depth=depth, normal=normal,
                    image_path=image_path, image_name=image_name, width=W1, height=H1, intrinsics=intrinsic, extrinsics=extrinsic))
            
    return cam_infos

def readCamerasFromDUST3Rtest(img_path, cam_path, white_background, extension=".jpg", scale = 50, train_image = None, dep_path = None, nor_path = None):
    #for zju competition, not know the img
    cam_infos = []
    image_files = sorted(os.listdir(img_path))
    cam_files = sorted(os.listdir(cam_path))
    idx = 0
    for cam_file in cam_files:
        image_name = os.path.splitext(cam_file)[0][:-4]
        idx += 1
        image_file = image_name + extension
        image_path = os.path.join(img_path, image_file)
        camera_path = os.path.join(cam_path, cam_file)

        if image_name in image_files:
            image = Image.open(image_path)
            W1, H1 = image.size
            im_data = np.array(image.convert("RGBA"))
            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")
        else:
            image = train_image
            W1, H1 = image.width, image.height
       
        depth = None
        normal = None

        with open(camera_path, 'r') as file:
            lines = file.readlines()

        c2w = []
        for i in range(1,5):
            line = lines[i].strip().split()
            row = [float(val) for val in line]
            c2w.append(row)
        c2w = np.array(c2w)
        w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3,:3])  
        T = w2c[:3, 3] * scale

        K = []
        for i in range(7, 10):
            line = lines[i].strip().split()
            row = [float(val) for val in line]
            K.append(row)
        K = np.array(K)
        W2, H2 = K[0][2]*2, K[1][2]*2  
        FovX = focal2fov(K[0,0],W2)
        FovY = focal2fov(K[1,1],H2)
        intrinsic = np.array([fov2focal(FovX, W1), fov2focal(FovY, H1), W1/2, H1/2])
        intrinsic = np.array([[fov2focal(FovX, W1), 0, W1/2],[0, fov2focal(FovY, H1), H1/2],[0,0,1]])
        extrinsic = np.eye(4)
        extrinsic[:3,:3] = w2c[:3,:3]
        extrinsic[:3,3] = T
        
        cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image, depth=depth, normal=normal,
                    image_path=image_path, image_name=image_name, width=W1, height=H1, intrinsics=intrinsic, extrinsics=extrinsic))
            
    return cam_infos

def readDUST3RInfo(path, white_background, depth, normal, eval, extension=".jpg"): 
    scale = 100  # dust3r scale is too small, 3dgs SIBR viewer cannot see, so we scale 100
    if os.path.exists(os.path.join(path, "train/cams")) and  os.path.exists(os.path.join(path, "train/images")):
        cams_folder = os.path.join(path, "train/cams")
        images_folder = os.path.join(path, "train/images")
        if depth:
            depth_folder = os.path.join(path, "train/depth_maps")
        else: 
            depth_folder = None
        if normal:
            normal_folder = os.path.join(path, "train/normal_maps")
        else:
            normal_folder = None
    else:
        raise Exception("Error message: no cams folder exits")
    
    train_cam_infos = readCamerasFromDUST3R(images_folder, cams_folder, white_background, extension, scale, depth_folder, normal_folder)

    if os.path.exists(os.path.join(path, "test/cams")) and  os.path.exists(os.path.join(path, "test/images")):
        cams_folder2 = os.path.join(path, "test/cams")
        images_folder2 = os.path.join(path, "test/images")
        print(f'-----test_image---------')
        test_cam_infos = readCamerasFromDUST3R(images_folder2, cams_folder2, white_background, extension, scale, None, None)
    else:
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "train/points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)


    try:
        pcd = fetchPly_scale(ply_path, scale)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readLLFFInfo(img_path, cam_path, white_background, extension=".jpg", scale = 50, dep_path = None, nor_path = None):
    cam_infos = []
    image_files = sorted(os.listdir(img_path))
    cam_files = sorted(os.listdir(cam_path))
    if dep_path != None:
        dep_files = sorted(os.listdir(dep_path))
    if nor_path != None:
        nor_files = sorted(os.listdir(nor_path))
    idx = 0
    for image_file in image_files:
        image_name = os.path.splitext(image_file)[0]
        #idx = int(image_name)
        idx += 1
        cam_file = f"{image_name}_cam.txt"
        dep_file = f"depth_{image_name}.png"
        nor_file = f"normal_{image_name}.png"
        if cam_file in cam_files:
            image_path = os.path.join(img_path, image_file)
            camera_path = os.path.join(cam_path, cam_file)
        else :
            raise Exception("Error message: no cam file exits matched with{image_file}")
        
        image = Image.open(image_path)
        W1, H1 = image.size
        W2, H2 = 512, 384
        im_data = np.array(image.convert("RGBA"))
        bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

        norm_data = im_data / 255.0
        arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
        image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

        if  dep_path != None and dep_file in dep_files:
            depth_path = os.path.join(dep_path, dep_file)
            depth = cv2.imread(depth_path, -1)
            depth = (depth / 1000).astype(np.float32) 
            depth = cv2.resize(depth, (W1, H1), interpolation=cv2.INTER_LINEAR)
        else: 
            depth = None

        if  nor_path != None and nor_file in nor_files:
            normal_path = os.path.join(nor_path, nor_file)
            normal = cv2.imread(normal_path, -1)
            normal = normal.astype(np.float32) 
            normal = cv2.resize(normal, (W1, H1), interpolation=cv2.INTER_LINEAR)
        else: 
            normal = None

        with open(camera_path, 'r') as file:
            lines = file.readlines()

        c2w = []
        for i in range(1,5):
            line = lines[i].strip().split()
            row = [float(val) for val in line]
            c2w.append(row)
        c2w = np.array(c2w)
        w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3,:3])  
        T = w2c[:3, 3] * scale

        K = []
        for i in range(7, 10):
            line = lines[i].strip().split()
            row = [float(val) for val in line]
            K.append(row)
        K = np.array(K)
        FovX = focal2fov(K[0,0],W2)
        FovY = focal2fov(K[1,1],H2)
        intrinsic = np.array([fov2focal(FovX, W1), fov2focal(FovY, H1), W1/2, H1/2])
        intrinsic = np.array([[fov2focal(FovX, W1), 0, W1/2],[0, fov2focal(FovY, H1), H1/2],[0,0,1]])
        extrinsic = np.eye(4)
        extrinsic[:3,:3] = w2c[:3,:3]
        extrinsic[:3,3] = T
        
        cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image, depth=depth, normal=normal,
                    image_path=image_path, image_name=image_name, width=W1, height=H1, intrinsics=intrinsic, extrinsics=extrinsic))
            
    return cam_infos

def readLLFFInfo(path, white_background, depth, normal, eval, extension=".jpg"): 
    scale = 100  # dust3r scale is too small!
    if os.path.exists(os.path.join(path, "train/cams")) and  os.path.exists(os.path.join(path, "train/images")):
        cams_folder = os.path.join(path, "train/cams")
        images_folder = os.path.join(path, "train/images")
        if depth:
            depth_folder = os.path.join(path, "train/depth_maps")
        else: 
            depth_folder = None
        if normal:
            normal_folder = os.path.join(path, "train/normal_maps")
        else:
            normal_folder = None
    else:
        raise Exception("Error message: no cams folder exits")
    
    train_cam_infos = readCamerasFromTankandTemples(images_folder, cams_folder, white_background, extension, scale, depth_folder, normal_folder)

    if os.path.exists(os.path.join(path, "test/cams")) and  os.path.exists(os.path.join(path, "test/images")):
        cams_folder2 = os.path.join(path, "test/cams")
        images_folder2 = os.path.join(path, "test/images")
        print(f'-----test_image---------')
        test_cam_infos = readCamerasFromTankandTemples(images_folder2, cams_folder2, white_background, extension, scale)
    else:
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "train/points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)


    try:
        pcd = fetchPly_scale(ply_path, scale)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo,
    "DUST3R": readDUST3RInfo,
    "LLFF" : readLLFFInfo,   
}