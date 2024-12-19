from diffusers import DDPMScheduler, UNet2DModel,FlowMatchEulerDiscreteScheduler,StableDiffusion3Pipeline,FluxPipeline,StableDiffusionPipeline,DDIMScheduler,DDIMInverseScheduler
from PIL import Image
import torch
from typing import Any, Callable, Dict, List, Optional, Union
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.utils.torch_utils import randn_tensor
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
import numpy as np

from utils import SD,set_ddim_timesteps,save_difference_image
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


# pipe = SD.from_pretrained("/mnt/workspace/common/models/stable-diffusion-v1-5", torch_dtype=torch.float16)
# pipe.set_ddim_scheduler("/mnt/workspace/common/models/stable-diffusion-v1-5", torch_dtype=torch.float16)

pipe = SD.from_pretrained("/mnt/workspace/common/models/stable-diffusion-2-1-base", torch_dtype=torch.float16)
pipe.set_ddim_scheduler("/mnt/workspace/common/models/stable-diffusion-2-1-base", torch_dtype=torch.float16)

pipe = pipe.to("cuda")


mse_list = []

for prompt in prompt_list:
    
    # 保证了所有图像都是pipe自己generation的
    image_ori = pipe(
        prompt,
        num_inference_steps=50,
    ).images[0]
    
    num_inference_steps = 50
    timesteps = set_ddim_timesteps(self = pipe.scheduler, num_inference_steps = num_inference_steps)

    timesteps = torch.flip(timesteps,dims=[0])

    latents_x0_to_noise,image_x0_to_noise = pipe.sd_x0_to_noise(
    prompt,
    num_inference_steps=num_inference_steps,
    timesteps = timesteps,
    image = image_ori,
    guidance_scale=1.0,
    )

    timesteps = torch.flip(timesteps,dims=[0])
    latents_noise_to_x0,image_noise_to_x0 = pipe.sd_noise_to_x0(
    prompt,
    num_inference_steps=num_inference_steps,
    timesteps = timesteps,
    # image = image,
    latents=latents_x0_to_noise,
    guidance_scale=1.0,
    )
    output_dir = "outputs/SD2.1/start_from_image"


    if not os.path.exists(output_dir):
        os.makedirs(output_dir)  

    output_path = os.path.join(output_dir, f"{prompt}_steps_{num_inference_steps}.png")
    mse = save_difference_image(image_ori, image_noise_to_x0, output_path)
    mse_list.append(mse)

average_mse = np.mean(mse_list)
print(average_mse)
# SD1.5,38.94772875467937
# SD2.1,28.45414655049642









