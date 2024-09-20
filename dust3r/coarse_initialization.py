from dust3r.inference import inference, load_model
from dust3r.utils.image import load_images
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from plyfile import PlyData, PlyElement
import os
import numpy as np
import cv2
import numpy as np
import torch
from dust3r.utils.device import to_numpy
from argparse import ArgumentParser

def uint8(colors):
    if not isinstance(colors, np.ndarray):
        colors = np.array(colors)
    if np.issubdtype(colors.dtype, np.floating):
        colors *= 255
    assert 0 <= colors.min() and colors.max() < 256
    return np.uint8(colors)

def save_point3dply_downsample(path: str, pts3d: list, mask: list, color: list, step: int = 4):
    pts3d = to_numpy(pts3d)
    mask = to_numpy(mask)
    pts = np.concatenate([p[m] for p, m in zip(pts3d, mask)])
    color = to_numpy(color)
    col = np.concatenate([p[m] for p, m in zip(color, mask)])
    num_samples = int(len(pts) / step) 
    random_indices = np.random.choice(len(pts), size=num_samples, replace=False)
    pts = pts[random_indices]
    col = col[random_indices]
    
    assert col.shape == pts.shape
    color = uint8(col.reshape(-1, 3))    
    num_points = col.shape[0]
    
    header = "ply\n"
    header += "format ascii 1.0\n"
    header += "element vertex {}\n".format(num_points)
    header += "property float x\n"
    header += "property float y\n"
    header += "property float z\n"
    header += "property float nx\n"
    header += "property float ny\n"
    header += "property float nz\n"   
    header += "property uchar red\n"
    header += "property uchar green\n"
    header += "property uchar blue\n"
    header += "end_header\n"
    
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")
    
    vertex = np.zeros(num_points, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                                         ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
                                         ('red', 'uint8'), ('green', 'uint8'), ('blue', 'uint8')])
    vertex['x'] = pts[:, 0]
    vertex['y'] = pts[:, 1]
    vertex['z'] = pts[:, 2]
    vertex['nx'] = 0
    vertex['ny'] = 0
    vertex['nz'] = 0
    vertex['red'] = col[:, 0]
    vertex['green'] = col[:, 1]
    vertex['blue'] = col[:, 2]

    vertex_element = PlyElement.describe(vertex, 'vertex')
    plydata = PlyData([vertex_element])
    plydata.write(path + '/points3d.ply')

    print(f"Saved points3d.ply to {path}")

def save_cams(path, poses, focal, img_filenames, cc):
    poses = to_numpy(poses)
    cx, cy = cc
    for i in range(len(img_filenames)):
        pose = poses[i]
        img_filename = img_filenames[i]
        txt_filename = img_filename.split('.')[0] + '_cam.txt'
        folder_path = path + '/cams/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        with open(folder_path + txt_filename, 'w') as f:
            f.write('extrinsic\n')
            for row in pose:
                line = ' '.join([str(val) for val in row])
                f.write(line + '\n')
            f.write('\nintrinsic\n')
            f.write(str(focal) + ' 0 ' + str(cx) + '\n')
            f.write('0 ' + str(focal)+ ' ' + str(cy) + '\n')
            f.write('0 0 1\n')

def save_depths_to_folder(folder_path, depths):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    for i, depth_map in enumerate(depths):
        depth_np = depth_map.clone().detach().cpu().numpy()
        normalized_depth =(depth_np - depth_np.min()) / (depth_np.max() - depth_np.min())
        colored_depth = cv2.cvtColor(normalized_depth*255, cv2.COLOR_GRAY2BGR).astype(np.uint8)
        colored_depth = cv2.applyColorMap(colored_depth, cv2.COLORMAP_JET)
        file_path = os.path.join(folder_path, f"depth_map_{i}.png")
        cv2.imwrite(file_path, colored_depth)
      

if __name__ == '__main__':
    os.chdir('./dust3r')
    parser = ArgumentParser(description="DUST3R intialization parameters")
    parser.add_argument('-s', type=str, help='Path to the data directory')
    args = parser.parse_args()

    model_path = "checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
    device = 'cuda'
    batch_size = 1
    schedule = 'cosine'
    lr = 0.01
    niter = 500
    print("get train poses and point clouds")
    model = load_model(model_path, device)

    dataset_path = '../' + args.s
    dataset_path_train = dataset_path + '/train'
    dest_train = dataset_path_train + '/images'
   
    print("incremental dust3r initialization")

    # the train
    images = load_images(dest_train, size=512)
    pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)
    output = inference(pairs, model, device, batch_size=batch_size)
    scene = global_aligner(output, device=device, mode=GlobalAlignerMode.ModularPointCloudOptimizer, depth_path = None) 
    loss = scene.compute_global_alignment(init="mst", niter=niter, schedule=schedule, lr=lr)
    scene.clean_pointcloud_final()
    depths = scene.get_depthmaps()
    save_depths_to_folder(dataset_path_train + '/dust3r_depth', depths)
    imgs0 = scene.imgs
    focals0 = scene.get_focals()
    poses0 = scene.get_im_poses()
    pts3ds0 = scene.get_pts3d()
    scene.min_conf_thr = float(scene.conf_trf(torch.tensor(5.0))) 
    confidence_masks0 = scene.get_masks_depth()   
    num_img0 = scene.n_imgs
    av_focal0 = (sum(focals0) / len(focals0)).item() 
    imgs_name = sorted(os.listdir(dest_train))
    cc = [imgs0[0].shape[1] / 2.0, imgs0[0].shape[0] / 2.0]
    save_cams(dataset_path_train, poses0, av_focal0, imgs_name, cc)
    save_point3dply_downsample(dataset_path_train, pts3ds0, confidence_masks0, imgs0, 1)



