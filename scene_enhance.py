import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video
from argparse import ArgumentParser
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from moviepy.editor import *


if __name__ == '__main__':
    parser = ArgumentParser(description="DUST3R intialization parameters")
    parser.add_argument('--model_path', type=str, default='./models/zeroscope_v2_XL', help='Path to the video generation model')
    parser.add_argument('--input_path', type=str, default='', help='Path to the rendered_video') 
    parser.add_argument('--output_path', type=str, default='', help='Path to the video generation model') 
    parser.add_argument('--resolution', type=float, default='1', help='Downsample ratio of each image(depend on the CUDA MEMORY)') 
    parser.add_argument('--batch_size', type=int, default='30')     
    args = parser.parse_args()
    if args.output_path == '':
        args.output_path = os.path.dirname(args.input_path) + '/enhance_result.mp4'

    # # load video generation model
    pipe = DiffusionPipeline.from_pretrained(args.model_path, torch_dtype=torch.float16)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    pipe.enable_vae_slicing()
    cap = cv2.VideoCapture(args.input_path)

    video = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width = frame_rgb.shape[:2]
        pil_frame = Image.fromarray(frame_rgb).resize((int(width/args.resolution), int(height/args.resolution)) )
        video.append(pil_frame)
    cap.release()

    video_frames_list = []
    total_frame = len(video)
    for i in range(0, total_frame, args.batch_size):
        video_frame = pipe(prompt = "real-world, sharp, high contrast, gradation", 
                            video=video[i: min(i+ args.batch_size, total_frame)], 
                            strength=0.2).frames
        video_frames_list.append(video_frame[0])

    combined_frames = np.concatenate(video_frames_list, axis=0)

    video_path = export_to_video(combined_frames, args.output_path)

    mv = VideoFileClip(video_path)
    mv2 = mv.fx(vfx.lum_contrast, lum=10, contrast=0.1)
    
    mv2.write_videofile(os.path.dirname(args.input_path) + '/enhance_result2.mp4')
