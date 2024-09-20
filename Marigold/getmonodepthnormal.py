import os
from argparse import ArgumentParser
from marigold import MarigoldPipeline
import logging
import numpy as np
import torch
from tqdm.auto import tqdm
from glob import glob
from PIL import Image
import cv2
import diffusers

if __name__ == '__main__':
    # Directories
    os.chdir('./Marigold')
    parser = ArgumentParser(description="Monocular depth/normal maps parameters")
    parser.add_argument('-s', type=str, help='Path to the data directory')
    args = parser.parse_args()
    input_dir = "../" +  args.s + "/train/images"
    output_dir_color = "../" +  args.s + "/train/depth_maps"
    output_dir_normal = "../" +  args.s + "/train/normal_maps"

    # monocular depth maps
    ckpt_path = 'checkpoint/marigold-depth-lcm-v1-0'
    pipe = MarigoldPipeline.from_pretrained(ckpt_path)
    pipe = pipe.to("cuda")
    EXTENSION_LIST = [".jpg", ".jpeg", ".png"]

    # Image list
    rgb_filename_list = glob(os.path.join(input_dir, "*"))
    rgb_filename_list = [
        f for f in rgb_filename_list if os.path.splitext(f)[1].lower() in EXTENSION_LIST
    ]
    rgb_filename_list = sorted(rgb_filename_list)

    # Create output folders
    os.makedirs(output_dir_color, exist_ok=True)
    os.makedirs(output_dir_normal, exist_ok=True)

    # Run Inference
    with torch.no_grad():
        for rgb_path in tqdm(rgb_filename_list, desc=f"Estimating depth", leave=True):
            # Read input image
            input_image = Image.open(rgb_path)

            # Predict depth
            pipeline_output = pipe(
                input_image,
            )

            depth_pred: np.ndarray = pipeline_output.depth_np
            depth_colored: Image.Image = pipeline_output.depth_colored

            # Save as npy
            rgb_name_base = os.path.splitext(os.path.basename(rgb_path))[0]
            pred_name_base = rgb_name_base


            # Colorize
            colored_save_path = os.path.join(
                output_dir_color, f"depth_{pred_name_base}.png"
            )
            if os.path.exists(colored_save_path):
                logging.warning(f"Existing file: '{colored_save_path}' will be overwritten")
            #depth_colored.save(colored_save_path)
            cv2.imwrite(colored_save_path, depth_pred*255)

    # monocular normal maps
    pipe = diffusers.MarigoldNormalsPipeline.from_pretrained(
        "checkpoint/marigold-normals-lcm-v0-1", variant="fp16", torch_dtype=torch.float16
    ).to("cuda")
    image_paths = glob(os.path.join(input_dir, "*"))
    for rgb_path in tqdm(rgb_filename_list, desc=f"Estimating normal", leave=True):
            # Read input image
            image = Image.open(rgb_path)
            normals = pipe(image)
            print(np.max(normals.prediction))
            print(np.min(normals.prediction))
            arr = ((normals.prediction + 1) * 127).clip(0, 255).astype(np.uint8)
            print(arr.shape)
            image = Image.fromarray(arr.squeeze())
            # 显示图像
            image.show()

            vis = pipe.image_processor.visualize_normals(normals.prediction)
            rgb_name_base = os.path.splitext(os.path.basename(rgb_path))[0]
            colored_save_path = os.path.join(
                output_dir_normal, f"normal_{rgb_name_base}.png"
            )
            image.save(colored_save_path)


