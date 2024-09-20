import os
import io
import torch
import numpy as np
import open3d as o3d
import torch.nn.functional as F
import cv2
import einops
import random
from argparse import ArgumentParser
from dataclasses import dataclass
from functools import partial
from PIL import Image
import clip
from gaussian_renderer import render
from scene import GaussianModel
from arguments import PipelineParams
from scene.gaussian_model import BasicPointCloud
from plyfile import PlyData, PlyElement
from torch import nn
from torchvision.transforms import ToPILImage, ToTensor
from torchvision.utils import save_image, make_grid
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS
from torchmetrics.image import PeakSignalNoiseRatio as PSNR, StructuralSimilarityIndexMeasure as SSIM#, LearnedPerceptualImagePatchSimilarity as LPIPS
from torchmetrics.functional.regression import pearson_corrcoef
from cldm.ddim_hacked import DDIMSampler
from annotator.util import resize_image, HWC3
from cldm.model import create_model, load_state_dict
from minlora import add_lora, LoRAParametrization



@torch.no_grad()
def process(
    model,
    ddim_sampler: DDIMSampler,
    input_image: np.ndarray,
    prompt: str,
    a_prompt: str = '',
    n_prompt: str = '',
    num_samples: int = 1,
    image_resolution: int = 512,
    ddim_steps: int = 50,
    guess_mode: bool = False,
    strength: float = 1.0,
    scale: float = 1.0,
    eta: float = 1.0,
    denoise_strength: float = 1.0
):
    input_image = HWC3(input_image)
    detected_map = input_image.copy()

    img = resize_image(input_image, image_resolution)
    H, W, C = img.shape

    detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)

    control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
    control = torch.stack([control for _ in range(num_samples)], dim=0)
    control = einops.rearrange(control, 'b h w c -> b c h w').clone()

    img = torch.from_numpy(img.copy()).float().cuda() / 127.0 - 1.0
    img = torch.stack([img for _ in range(num_samples)], dim=0)
    img = einops.rearrange(img, 'b h w c -> b c h w').clone()

    cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
    un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}

    ddim_sampler.make_schedule(ddim_steps, ddim_eta=eta, verbose=False)
    t_enc = min(int(denoise_strength * ddim_steps), ddim_steps - 1)
    z = model.get_first_stage_encoding(model.encode_first_stage(img))
    z_enc = ddim_sampler.stochastic_encode(z, torch.tensor([t_enc] * num_samples).to(model.device))

    model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)
    # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01

    samples = ddim_sampler.decode(z_enc, cond, t_enc, unconditional_guidance_scale=scale, unconditional_conditioning=un_cond)

    x_samples = model.decode_first_stage(samples)
    x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

    results = [x_samples[i] for i in range(num_samples)]


    alphas = ddim_sampler.alphas_cumprod.cuda()
    sds_w = (1 - alphas[t_enc]).view(-1, 1)

    return results, sds_w