from diffusers import DDPMScheduler, UNet2DModel,FlowMatchEulerDiscreteScheduler,StableDiffusion3Pipeline,FluxPipeline,StableDiffusionPipeline,DDIMScheduler,DDIMInverseScheduler
from PIL import Image
import torch
from typing import Any, Callable, Dict, List, Optional, Union
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.utils.torch_utils import randn_tensor
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
import numpy as np

from utils import SD3,set_ddim_timesteps,set_flow_timesteps,save_difference_image
import os


prompt_list = [
    "A fantasy landscape at sunset",
    "A futuristic city skyline",
    "A dragon flying over mountains",
    "A serene beach with palm trees",
    "A steampunk robot in a workshop",
    "A magical forest with glowing creatures",
    "A portrait of a medieval knight",
    "A cyberpunk street scene at night",
    "A cute kitten playing with yarn",
    "A vibrant underwater coral reef",
    "A majestic castle on a hill",
    "A cozy cabin in the snow",
    "A spaceship exploring distant galaxies",
    "A vintage car on a country road",
    "A surreal dreamscape with floating islands",
    "A close-up of a flower in bloom",
    "A whimsical tea party in a garden",
    "A dark castle under a full moon",
    "A peaceful waterfall in a forest",
    "A busy market in an ancient city",
    "A majestic eagle soaring in the sky",
    "A portrait of a famous historical figure",
    "A cozy reading nook by a window",
    "A dramatic storm over a rocky coast",
    "A colorful festival parade",
    "A sleek futuristic vehicle on the road",
    "A serene mountain lake at dawn",
    "A retro diner in the 1950s",
    "A tranquil zen garden with cherry blossoms",
    "A vibrant carnival scene",
    "A mysterious cave with glowing crystals",
    "A bustling city street during rush hour",
    "A magical library filled with ancient books",
    "A close-up of a butterfly on a flower",
    "A retro sci-fi movie poster",
    "A spooky haunted house",
    "A scenic vineyard in autumn",
    "A festive holiday celebration",
    "A charming European village",
    "A powerful thunderstorm with lightning",
    "A dramatic cliffside view of the ocean",
    "A tranquil river winding through a valley",
    "A stylish fashion model on a runway",
    "A playful puppy in a park",
    "A vivid sunset over a cityscape",
    "A cozy fireplace with stockings",
    "A vintage airplane flying in the clouds",
    "A festive winter wonderland",
    "A detailed map of a fictional world",
    "A whimsical creature in a magical realm"
]


pipe = SD3.from_pretrained("/mnt/workspace/common/models/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16)
pipe = pipe.to("cuda")

mse_list = []
num_inference_steps = 50

for prompt in prompt_list:
    print(prompt)
    # 保证了所有图像都是pipe自己generation的
    image_ori = pipe(
        prompt,
        negative_prompt="",
        num_inference_steps=28,
        guidance_scale=7.0,
        height=768,
        width=1360
    ).images[0]
    output_path = "outputs/SD3/generation/"
    if not os.path.exists(os.path.dirname(output_path)):
        os.mkdir(os.path.dirname(output_path))
    output_path = os.path.join(output_path,prompt+".png")
    image_ori.save(output_path,format="PNG")


    










