from diffusers import DDPMScheduler, UNet2DModel,FlowMatchEulerDiscreteScheduler,StableDiffusion3Pipeline,FluxPipeline,StableDiffusionPipeline,DDIMScheduler,DDIMInverseScheduler
from PIL import Image
import torch
from typing import Any, Callable, Dict, List, Optional, Union
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.utils.torch_utils import randn_tensor
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
import numpy as np

from utils import Flux,set_ddim_timesteps,set_flow_timesteps,save_difference_image,get_custom_timesteps,save_difference_image_all
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


pipe = Flux.from_pretrained("/mnt/workspace/common/models/FLUX.1-dev", torch_dtype=torch.bfloat16)
# pipe.enable_model_cpu_offload()
pipe = pipe.to("cuda")

mse_list = []
mse_custom_list = []
num_inference_steps = 30
guidance_scale = 2.0
height = 768
width = 1360


for prompt in prompt_list:
    
    # 保证了所有图像都是pipe自己generation的
    image_ori = pipe(
        prompt=prompt,
        guidance_scale=3.5,
        height=height,
        width=width,
        num_inference_steps=50,
    ).images[0]
    
    timesteps = set_flow_timesteps(self = pipe.scheduler, num_inference_steps = num_inference_steps)
    timesteps = torch.flip(timesteps,dims=[0])
    print(timesteps)

    latents_x0_to_noise,image_x0_to_noise = pipe.flux_x0_to_noise(
    prompt,
    num_inference_steps=num_inference_steps,
    timesteps = timesteps,
    image = image_ori,
    guidance_scale=guidance_scale,
    height=height,
    width=width,
    )

    timesteps = torch.flip(timesteps,dims=[0])
    print(timesteps)

    latents_noise_to_x0,image_noise_to_x0 = pipe.flux_noise_to_x0(
    prompt,
    num_inference_steps=num_inference_steps,
    timesteps = timesteps,
    # image = image,
    latents=latents_x0_to_noise,
    guidance_scale=guidance_scale,
    height=height,
    width=width,
    )


    timesteps = get_custom_timesteps(num_inference_steps)
    print(timesteps)

    latents_x0_to_noise_custom,image_x0_to_noise_custom = pipe.flux_x0_to_noise(
    prompt,
    num_inference_steps=num_inference_steps,
    timesteps = timesteps,
    image = image_ori,
    guidance_scale=guidance_scale,
    height=height,
    width=width,
    )

    timesteps = torch.flip(timesteps,dims=[0])
    print(timesteps)

    latents_noise_to_x0_custom, image_noise_to_x0_custom = pipe.flux_noise_to_x0(
    prompt,
    num_inference_steps=num_inference_steps,
    timesteps = timesteps,
    # image = image,
    latents=latents_x0_to_noise_custom,
    guidance_scale=guidance_scale,
    height=height,
    width=width,
    )



    output_dir = f"outputs/Flux/start_from_image_custom_lin_inferences_{num_inference_steps}_cfg_{guidance_scale}_res_{height}_{width}"


    if not os.path.exists(output_dir):
        os.makedirs(output_dir)  

    output_path = os.path.join(output_dir, f"{prompt}_steps_{num_inference_steps}.png")
    
    mse,mse_custom = save_difference_image_all(image_ori, image_noise_to_x0,image_noise_to_x0_custom, output_path)
    # mse_custom = save_difference_image(image_ori, image_noise_to_x0, output_path)

    mse_list.append(mse)
    mse_custom_list.append(mse_custom)
average_mse = np.mean(mse_list)
average_mse_custom = np.mean(mse_custom_list)
print(average_mse)
print(average_mse_custom)
# Flux 37.97842694600423 # 50steps



# height = 768
# width = 1360
# steps = 30
#66.02855369817198
#82.17937082567401









