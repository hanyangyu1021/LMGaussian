import argparse
import os
import torch
import pytorch_lightning as pl
from functools import partial
from torch import nn
from pytorch_lightning.loggers import TensorBoardLogger
from cldm.model import create_model, load_state_dict
from cldm.logger import ImageLogger, LoraCheckpoint
from torch.utils.data import DataLoader
from minlora import add_lora, LoRAParametrization
import pickle
import numpy as np
from random import choice
from typing import List, Union
from PIL import Image
from argparse import ArgumentParser
from torch.utils.data import Dataset
from gaussian_renderer import render
from scene import GaussianModel
from arguments import PipelineParams
from scene.cameras import Camera
from utils.graphics_utils import focal2fov
import cv2

def load_statistics_info(info_path):
    with open(info_path, "rb") as f:
        info = pickle.load(f)
    return info

class GSDataset(Dataset):
    def __init__(
        self,
        gaussian_dir: str,
        data_dir: str,
        image_size: int = 512,
        resolution: int = 4,
        noise_scale_min: float = 0.6,
        noise_scale_max: float = 0.8,
        noise_dropout_min: float = 0.6,
        noise_dropout_max: float = 0.8,
        manual_noise_reduce_start: int = 20,
        manual_noise_reduce_gamma: float = 0.98,
        prompt: str = '',
        bg_white: bool = False,
        sh_degree: int = 3,
        use_prompt_list: bool = False,
        cache_max_iter: int = 100,
        train: bool = True
    ):
        super().__init__()
        self.gaussian_dir = gaussian_dir
        self.data_dir = data_dir
        self.image_size = image_size
        self.resolution = resolution
        self.noise_scale_min = noise_scale_min
        self.noise_scale_max = noise_scale_max
        self.noise_dropout_min = noise_dropout_min
        self.noise_dropout_max = noise_dropout_max
        self.current_step = 0
        self.manual_noise_prob = 0
        self.manual_noise_reduce_start = manual_noise_reduce_start
        self.manual_noise_reduce_gamma = manual_noise_reduce_gamma
        self.prompt = prompt
        self.train = train
        self.bg_white = bg_white
        self.bg_color = torch.tensor([1., 1., 1.] if bg_white else [0., 0., 0.] , dtype=torch.float32, device='cuda')
        self.use_prompt_list = use_prompt_list
        self.cache_max_iter = cache_max_iter

        self.iter = max([int(iter.split('_')[-1]) for iter in os.listdir(os.path.join(self.gaussian_dir, 'point_cloud'))
                         if os.path.isdir(os.path.join(self.gaussian_dir, 'point_cloud', iter)) and iter.split('_')[-1].isdigit()])

        ply_path = os.path.join(self.gaussian_dir, 'point_cloud', f'iteration_{self.iter}', 'point_cloud.ply')
        self.gaussian = GaussianModel(sh_degree=sh_degree)
        self.gaussian.load_ply(ply_path)
        self.parser = ArgumentParser(description="Training script parameters")
        self.pipe = PipelineParams(self.parser)

        path = self.data_dir
        if os.path.exists(os.path.join(path, "train/cams")) and  os.path.exists(os.path.join(path, "train/images")):
            cam_path = os.path.join(path, "train/cams")
            img_path = os.path.join(path, "train/images")
        else:
            raise Exception("Error message: no cams folder exits")
        
        self.Rs: List[np.ndarray] = []
        self.Ts: List[np.ndarray] = []
        self.heights: List[float] = []
        self.widths: List[float] = []
        self.fovxs: List[float] = []
        self.fovys: List[float] = []
        self.images: List[np.ndarray] = []
        self.noisys: List[List[np.ndarray]] = []

        self.statistics_info = []
        image_files = sorted(os.listdir(img_path))
        cam_files = sorted(os.listdir(cam_path))
        idx = 0
        for image_file in image_files:
            image_name = os.path.splitext(image_file)[0]
            cam_file = f"{image_name}_cam.txt"
            if cam_file in cam_files:
                image_path = os.path.join(img_path, image_file)
                camera_path = os.path.join(cam_path, cam_file)
            else :
                raise Exception("Error message: no cam file exits matched with{image_file}")
            
            image = Image.open(image_path)
            W1, H1 = image.size
            image = np.array(image.convert("RGB"))
        
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
            T = w2c[:3, 3] * 100 #scale set before

            K = []
            for i in range(7, 10):
                line = lines[i].strip().split()
                row = [float(val) for val in line]
                K.append(row)
            K = np.array(K)
            W2, H2 = K[0][2]*2, K[1][2]*2  
            FovX = focal2fov(K[0,0],W2)
            FovY = focal2fov(K[1,1],H2)

            self.Rs.append(R)
            self.Ts.append(T)
            self.heights.append(H1)
            self.widths.append(W1)
            self.fovxs.append(FovX)
            self.fovys.append(FovY)
            self.images.append(image)

            if self.train:
                    noisy_paths = os.listdir(os.path.join(self.gaussian_dir, f'{idx}'))
                    its = sorted([int(path.replace('sample_', '').replace('.png', '')) for path in noisy_paths])
                    noisys = [np.array(Image.open(os.path.join(self.gaussian_dir, f'{idx}', f'sample_{it}.png'))) for it in its[:self.cache_max_iter]]
                    self.noisys.append(noisys)
                    print(f'Load {len(noisys)} images for {idx}')
                    
            idx += 1


    def __len__(self):
        return len(self.images)

    def center_pad(self, image: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        H, W, _ = image.shape
        min_side = min(H, W)
        scale_factor = self.image_size / min_side
        resized_image = None
        if isinstance(image, torch.Tensor):
            resized_image = torch.nn.functional.interpolate(image.unsqueeze(0), scale_factor=scale_factor, mode='bilinear', align_corners=False)
        else:
            pil_image = Image.fromarray(np.uint8(image * 255.0 ))
            resized_pil_image = pil_image.resize((int(W * scale_factor), int(H * scale_factor)), resample=Image.BILINEAR)
            resized_image = np.array(resized_pil_image) / 255.0

        H_resized, W_resized, _ = resized_image.shape
        pad_l = max((self.image_size - W_resized) // 2, 0)
        pad_r = max(self.image_size - W_resized - pad_l, 0)
        pad_u = max((self.image_size - H_resized) // 2, 0)
        pad_d = max(self.image_size - H_resized - pad_u, 0)
        if isinstance(image, torch.Tensor):
            return torch.nn.functional.pad(resized_image, (0, 0, pad_l, pad_r, pad_u, pad_d), mode='constant', value=1. if self.bg_white else 0.)
        else:
            return np.pad(resized_image, ((pad_u, pad_d), (pad_l, pad_r), (0, 0)), mode='constant', constant_values=1. if self.bg_white else 0.)

    def center_crop(self, image: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        image = self.center_pad(image)
        H, W, _ = image.shape
        up = H // 2 - self.image_size // 2
        down = up + self.image_size
        left = W // 2 - self.image_size // 2
        right = left + self.image_size
        return image[up:down, left:right, :]
    
    def random_crop1(self, image: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        image = self.center_pad(image)
        H, W, _ = image.shape
        up = H // 2 - self.image_size // 2
        down = up + self.image_size
        left = np.uint8(np.random.normal(W // 2 - self.image_size // 2, W // 6 - self.image_size // 6))
        left = min(left, W - self.image_size)
        right = left + self.image_size
        return image[up:down, left:right, :], left

    def random_crop2(self, image: Union[np.ndarray, torch.Tensor], left: np.uint8):
        image = self.center_pad(image)
        H, W, _ = image.shape
        up = H // 2 - self.image_size // 2
        down = up + self.image_size
        right = left + self.image_size
        return image[up:down, left:right, :]

    def resize_image(self, input_image: np.ndarray) -> np.ndarray:
        H, W, _ = input_image.shape
        H = int(np.round(H / 64.0)) * 64
        W = int(np.round(W / 64.0)) * 64
        img = cv2.resize(input_image, (W, H), interpolation=cv2.INTER_AREA)
        return img

    @torch.no_grad()
    def __getitem__(self, idx):
        image = self.images[idx]
        cam = Camera(
            colmap_id = 0,
            R = self.Rs[idx],
            T = self.Ts[idx],
            FoVx = self.fovxs[idx],
            FoVy = self.fovys[idx],
            image = torch.from_numpy(self.images[idx].astype(np.float32) / 255.0).permute(2, 0, 1),
            gt_alpha_mask = None,
            gt_depth = None,
            gt_normal= None,
            image_name = 'tmp.png',
            uid = 0
        )

        if not self.train:
            render_pkg = render(cam, self.gaussian, self.pipe, self.bg_color)
            noisy = render_pkg['render']
            source = self.center_crop(noisy.clamp(0., 1.).permute(1, 2, 0))
        else:
            noisy = choice(self.noisys[idx])
            source, left = self.random_crop1(noisy.astype(np.float32) / 255)
           
        target = self.random_crop2(image.astype(np.float32) / 255, left) *2 - 1.0

        self.current_step += 1
        if self.current_step >= self.manual_noise_reduce_start:
            self.manual_noise_prob = self.manual_noise_prob * self.manual_noise_reduce_gamma

        if target.shape != (512, 512, 3):
            print("stop\n")

        return {
            'jpg': target,
            'txt': f'{self.prompt}',
            'hint': source
        }




_ = torch.set_grad_enabled(False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process experiment parameters.')

    parser.add_argument('--model_name', type=str, default='control_v11f1e_sd15_tile')
    parser.add_argument('--sh_degree', type=int, default=3)
    parser.add_argument('--image_size', type=int, default=512)
    parser.add_argument('--resolution', type=int, default=1)
    parser.add_argument('--gs_dir', type=str, default=f'output/horse16')
    parser.add_argument('--data_dir', type=str, default=f'data/horse16')
    parser.add_argument('--prompt', type=str, default='high quality, sharp outside scene')
    parser.add_argument('--exp_name', type=str, default=f'controlnet_finetune/horse16')
    parser.add_argument('--bg_white', action='store_true', default=False)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--only_mid_control', action='store_true', default=False)
    parser.add_argument('--lora_rank', type=int, default=64)
    parser.add_argument('--callbacks_every_n_train_steps', type=int, default=600)
    parser.add_argument('--max_steps', type=int, default=1800)
    parser.add_argument('--use_prompt_list', action='store_true', default=False)
    parser.add_argument('--manual_noise_reduce_start', type=int, default=100)
    parser.add_argument('--manual_noise_reduce_gamma', type=float, default=0.995)
    parser.add_argument('--cache_max_iter', type=int, default=50)


    args = parser.parse_args()

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



    exp_path = args.exp_name
    dataset = GSDataset(
        args.gs_dir, args.data_dir, 
        prompt=args.prompt,
        bg_white=args.bg_white,
        train=True,
        manual_noise_reduce_gamma=args.manual_noise_reduce_gamma,
        manual_noise_reduce_start=args.manual_noise_reduce_start,
        sh_degree=args.sh_degree,
        image_size=args.image_size,
        resolution=args.resolution,
        use_prompt_list=args.use_prompt_list,
        cache_max_iter=args.cache_max_iter
    )
    dataloader = DataLoader(dataset, num_workers=0, batch_size=args.batch_size, shuffle=True)
    loggers = [
        TensorBoardLogger(os.path.join(exp_path, 'tf_logs'))
    ]
    callbacks = [
        ImageLogger(exp_dir=exp_path, every_n_train_steps=args.callbacks_every_n_train_steps, \
                    log_images_kwargs = {"plot_diffusion_rows": True, "sample": True}),
        LoraCheckpoint(exp_dir=exp_path, every_n_train_steps=args.callbacks_every_n_train_steps)
    ]
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=1,
        precision=32,
        logger=loggers,
        callbacks=callbacks,
        max_steps=args.max_steps,
        check_val_every_n_epoch=args.callbacks_every_n_train_steps//len(dataset)*2
    )

    trainer.fit(model, dataloader)
