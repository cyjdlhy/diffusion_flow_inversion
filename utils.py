from diffusers import DDPMScheduler, UNet2DModel,FlowMatchEulerDiscreteScheduler,StableDiffusion3Pipeline,FluxPipeline,StableDiffusionPipeline,DDIMScheduler,DDIMInverseScheduler
from PIL import Image
import torch
from typing import Any, Callable, Dict, List, Optional, Union
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.utils.torch_utils import randn_tensor
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
import numpy as np
from utils import *
import cv2
import matplotlib.pyplot as plt
def save_difference_image(image1_path,image2_path,output_path):
    # 读取图像
    if isinstance(image1_path,str): 
        img1 = cv2.imread(image1_path)
    else:
        img1 = np.array(image1_path)
    
    if isinstance(image2_path,str):
        img2 = c2.imread(image2_path)
    else:
        img2 = np.array(image2_path)

    # img2 = cv2.imread(image2_path)
    print(img1.shape,img2.shape)
    # 确保两个图像具有相同的尺寸
    if img1.shape != img2.shape:
        print(img1.shape,img2.shape)
        raise ValueError("Images must have the same dimensions")

    # 计算MSE
    mse = np.mean((img1 - img2) ** 2)

    # 计算差异图像并归一化以便可视化
    diff_img = cv2.absdiff(img1, img2)
    diff_img = cv2.normalize(diff_img, None, 0, 255, cv2.NORM_MINMAX)

    # 打印MSE值
    print(f'MSE: {mse}')

    # 显示原始图像和差异图像
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    plt.title('Image 1')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    plt.title('Image 2')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(diff_img, cv2.COLOR_BGR2RGB))
    plt.title('Difference Image')
    plt.axis('off')

    plt.suptitle(f'MSE: {mse:.2f}')

    # 保存图片
    plt.savefig(output_path)
    plt.close()  # 关闭图像以释放内存

    return mse

import cv2
import numpy as np
import matplotlib.pyplot as plt

def save_difference_image_all(image1_path, image2_path, image3_path, output_path):
    # 读取图像
    def read_image(image_path):
        if isinstance(image_path, str):
            return cv2.imread(image_path)
        else:
            return np.array(image_path)
    
    img1 = read_image(image1_path)
    img2 = read_image(image2_path)
    img3 = read_image(image3_path)

    # 确保所有图像具有相同的尺寸
    if img1.shape != img2.shape or img1.shape != img3.shape:
        print(img1.shape, img2.shape, img3.shape)
        raise ValueError("All images must have the same dimensions")

    # 计算MSE函数
    def calculate_mse(img1, img2):
        return np.mean((img1 - img2) ** 2)

    # 计算差异图像并归一化
    def get_difference_image(img1, img2):
        diff_img = cv2.absdiff(img1, img2)
        return cv2.normalize(diff_img, None, 0, 255, cv2.NORM_MINMAX)

    # 计算 MSE 和差异图像
    mse_1_2 = calculate_mse(img1, img2)
    mse_1_3 = calculate_mse(img1, img3)
    
    diff_img_1_2 = get_difference_image(img1, img2)
    diff_img_1_3 = get_difference_image(img1, img3)

    # 打印 MSE 值
    print(f'MSE between Image 1 and Image 2: {mse_1_2}')
    print(f'MSE between Image 1 and Image 3: {mse_1_3}')

    # 显示原始图像和差异图像
    plt.figure(figsize=(15, 5))
    
    # 显示 image1 和 image2 的差异
    plt.subplot(2, 3, 1)
    plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    plt.title('Image 1')
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    plt.title('Image 2')
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.imshow(cv2.cvtColor(diff_img_1_2, cv2.COLOR_BGR2RGB))
    plt.title(f'Difference 1-2\nMSE: {mse_1_2:.2f}')
    plt.axis('off')

    # 显示 image1 和 image3 的差异
    plt.subplot(2, 3, 4)
    plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    plt.title('Image 1')
    plt.axis('off')

    plt.subplot(2, 3, 5)
    plt.imshow(cv2.cvtColor(img3, cv2.COLOR_BGR2RGB))
    plt.title('Image 3')
    plt.axis('off')

    plt.subplot(2, 3, 6)
    plt.imshow(cv2.cvtColor(diff_img_1_3, cv2.COLOR_BGR2RGB))
    plt.title(f'Difference 1-3\nMSE: {mse_1_3:.2f}')
    plt.axis('off')

    # 保存图片
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()  # 关闭图像以释放内存

    return mse_1_2, mse_1_3

        
def retrieve_latents(
    encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")

class SD3(StableDiffusion3Pipeline):
    

    @torch.no_grad()
    def ODE_step(self, latents, resolution, inverse_t, inverse_text_embeddings, inverse_guidance_scale, B, K = 0, weights=None, text_embeddings=None):
        
        with torch.no_grad():
            if inverse_guidance_scale > 1.0:

                latents_noisy = latents
                latent_model_input = latents_noisy[None, :, ...].repeat(2, 1, 1, 1, 1).reshape(-1, 16, resolution[0] // 8, resolution[1] // 8, )
                tt = torch.tensor([inverse_t]).reshape(1, 1).repeat(latent_model_input.shape[0], 1).reshape(-1)
                tt = tt.to(self.device)
                # 这里的self.transformer有两个embedding
                unet_output = self.transformer(hidden_states = latent_model_input.to(self.precision_t), 
                                            timestep = tt.to(self.precision_t), 
                                            encoder_hidden_states= inverse_text_embeddings[0].to(self.precision_t),
                                            pooled_projections = inverse_text_embeddings[1].to(self.precision_t),
                                            return_dict=False,
                                                )[0].to(latent_model_input.dtype)

                unet_output = unet_output.reshape(2, -1, 16, resolution[0] // 8, resolution[1] // 8, )
                noise_pred_uncond, noise_pred_text = unet_output[:1].reshape(-1, 16, resolution[0] // 8, resolution[1] // 8, ), unet_output[1:].reshape(-1, 16, resolution[0] // 8, resolution[1] // 8, )
                delta_RFDS = noise_pred_text - noise_pred_uncond
                pred_grad = noise_pred_uncond + inverse_guidance_scale * delta_RFDS
            
            else:
                
                latents_noisy = latents
                latent_model_input = latents_noisy[None, :, ...].repeat(1, 1, 1, 1, 1).reshape(-1, 16, resolution[0] // 8, resolution[1] // 8, )
                tt = torch.tensor([inverse_t]).reshape(1, 1).repeat(latent_model_input.shape[0], 1).reshape(-1)
                tt = tt.to(self.device)

                pred_grad = self.transformer(hidden_states = latent_model_input.to(self.precision_t), 
                                            timestep = tt.to(self.precision_t), 
                                            encoder_hidden_states= inverse_text_embeddings[0][B:].to(self.precision_t),
                                            pooled_projections = inverse_text_embeddings[1][B:].to(self.precision_t),
                                            return_dict=False,
                                                )[0].to(latent_model_input.dtype)
            return pred_grad


    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.retrieve_latents
    
    # from image to image
    def prepare_x0_latents(self, image, timestep, batch_size, num_images_per_prompt, dtype, device, generator=None):
        if not isinstance(image, (torch.Tensor, Image.Image, list)):
            raise ValueError(
                f"`image` has to be of type `torch.Tensor`, `PIL.Image.Image` or list but is {type(image)}"
            )

        image = image.to(device=device, dtype=dtype)

        batch_size = batch_size * num_images_per_prompt
        if image.shape[1] == self.vae.config.latent_channels:
            init_latents = image

        else:
            if isinstance(generator, list) and len(generator) != batch_size:
                raise ValueError(
                    f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                    f" size of {batch_size}. Make sure the batch size matches the length of the generators."
                )

            elif isinstance(generator, list):
                init_latents = [
                    retrieve_latents(self.vae.encode(image[i : i + 1]), generator=generator[i])
                    for i in range(batch_size)
                ]
                init_latents = torch.cat(init_latents, dim=0)
            else:
                init_latents = retrieve_latents(self.vae.encode(image), generator=generator)

            init_latents = (init_latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor

        if batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] == 0:
            # expand init_latents for batch_size
            additional_image_per_prompt = batch_size // init_latents.shape[0]
            init_latents = torch.cat([init_latents] * additional_image_per_prompt, dim=0)
        elif batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] != 0:
            raise ValueError(
                f"Cannot duplicate `image` of batch size {init_latents.shape[0]} to {batch_size} text prompts."
            )
        else:
            init_latents = torch.cat([init_latents], dim=0)

        if timestep is not None:
            shape = init_latents.shape
            noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)

            # get latents
            init_latents = self.scheduler.scale_noise(init_latents, timestep, noise)
        else:
            latents = init_latents.to(device=device, dtype=dtype)

        return latents



    # noise->x0
    @torch.no_grad()
    def sd3_noise_to_x0(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        prompt_3: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 28,
        timesteps: List[int] = None, # 我们可以自定义输入
        guidance_scale: float = 7.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt_3: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None, # 这个就是我们需要获得的latents
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 256,
    ):

        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor
        self.check_inputs(
            prompt,
            prompt_2,
            prompt_3,
            height,
            width,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            negative_prompt_3=negative_prompt_3,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._clip_skip = clip_skip
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        lora_scale = (
            self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None
        )
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_3=prompt_3,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            negative_prompt_3=negative_prompt_3,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            device=device,
            clip_skip=self.clip_skip,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )

        if self.do_classifier_free_guidance:
            print("will use guidance_scale")
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)

        if timesteps is None: # timesteps 是一个包括了起点和终点的一个timesteps
            raise ValueError("timesteps should not be None")
        else:
            # print(timesteps)
            pass
            
        # 5. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels
        if latents is None:
            # raise ValueError("latents should not be None")
            latents = self.prepare_latents(
                batch_size * num_images_per_prompt,
                num_channels_latents,
                height,
                width,
                prompt_embeds.dtype,
                device,
                generator,
                latents,
            )
        

        num_inference_steps = len(timesteps)-1 # 我们直接根据放入的timesteps 来决定怎么loop



        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i,t in enumerate(timesteps[:-1]):
                if self.interrupt:
                    continue
                
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                t = torch.tensor([min(t,1000.0)], device=self.device) # 限制在0~1000之间
                timestep = t.expand(latent_model_input.shape[0])

                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype

                dt = (timesteps[i+1]-t)/1000
                dt = torch.tensor(dt).to(latents_dtype).to(device)
                print(dt)
                latents = latents+ noise_pred * dt


                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)
                    negative_pooled_prompt_embeds = callback_outputs.pop(
                        "negative_pooled_prompt_embeds", negative_pooled_prompt_embeds
                    )

                # call the callback, if provided
                # if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                progress_bar.update()

                # if XLA_AVAILABLE:
                #     xm.mark_step()

    # if output_type == "latent":
        ori_latents = latents.clone()

    # else:
        latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        image = self.vae.decode(latents, return_dict=False)[0]
        image = self.image_processor.postprocess(image, output_type=output_type)
        
        return ori_latents,image[0]


    # x0->noise
    @torch.no_grad()
    def sd3_x0_to_noise(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        prompt_3: Optional[Union[str, List[str]]] = None,
        # height: Optional[int] = None,
        # width: Optional[int] = None,
        image: PipelineImageInput = None,
        strength: float = 0.6,
        num_inference_steps: int = 28,
        timesteps: List[int] = None, # 我们可以自定义输入
        guidance_scale: float = 7.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt_3: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None, # 这个就是我们需要获得的latents
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 256,
    ):

        # 1. Check inputs. Raise error if not correct
        # self.check_inputs(
        #     prompt,
        #     prompt_2,
        #     prompt_3,
        #     strength,
        #     negative_prompt=negative_prompt,
        #     negative_prompt_2=negative_prompt_2,
        #     negative_prompt_3=negative_prompt_3,
        #     prompt_embeds=prompt_embeds,
        #     negative_prompt_embeds=negative_prompt_embeds,
        #     pooled_prompt_embeds=pooled_prompt_embeds,
        #     negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        #     callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        #     max_sequence_length=max_sequence_length,
        # )

        self._guidance_scale = guidance_scale
        self._clip_skip = clip_skip
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        lora_scale = (
            self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None
        )
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_3=prompt_3,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            negative_prompt_3=negative_prompt_3,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            device=device,
            clip_skip=self.clip_skip,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )

        if self.do_classifier_free_guidance:
            print("will use guidance_scale")
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)

        # 3. Preprocess image
        image = self.image_processor.preprocess(image)

        # 5. Prepare latent variables
        latent_timestep = None # 这个是一种策略 非常关键
        if latents is None: # 这里latents 必须要None
            latents = self.prepare_x0_latents(
                image,
                latent_timestep,
                batch_size,
                num_images_per_prompt,
                prompt_embeds.dtype,
                device,
                generator,
            )


        if timesteps is None: # timesteps 是一个包括了起点和终点的一个timesteps
            raise ValueError("timesteps should not be None")
        else:
            # print(timesteps)
            pass
            
        # 5. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels
        if latents is None:
            raise ValueError("latents should not be None")

        num_inference_steps = len(timesteps)-1 # 我们直接根据放入的timesteps 来决定怎么loop



        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i,t in enumerate(timesteps[:-1]):
                if self.interrupt:
                    continue
                
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                t = torch.tensor([min(t,1000.0)], device=self.device) # 限制在0~1000之间
                timestep = t.expand(latent_model_input.shape[0])

                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype

                dt = (timesteps[i+1]-t)/1000
                dt = torch.tensor(dt).to(latents_dtype).to(device)
                print(dt)
                latents = latents+ noise_pred * dt

                # 下面几个if 不知道是干什么的
                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)
                    negative_pooled_prompt_embeds = callback_outputs.pop(
                        "negative_pooled_prompt_embeds", negative_pooled_prompt_embeds
                    )

                # call the callback, if provided
                # if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                progress_bar.update()

                # if XLA_AVAILABLE:
                #     xm.mark_step()


    # if output_type == "latent":
        ori_latents = latents.clone()

    # else:
        latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        image = self.vae.decode(latents, return_dict=False)[0]
        image = self.image_processor.postprocess(image, output_type=output_type)
          
        return ori_latents,image[0]


    @torch.no_grad()
    def get_latents(self,batch_size=1,num_images_per_prompt=1,num_channels_latents=16,height=512,
    width=512, dtype=torch.float16, device="cuda",generator=None,latents=None):
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            dtype,
            device,
            generator,
            latents,
        )
        output_type="pil"
    # if output_type == "latent":
        ori_latents = latents.clone()
    # else:
        latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        image = self.vae.decode(latents, return_dict=False)[0]
        image = self.image_processor.postprocess(image, output_type=output_type)
        
        return ori_latents,image[0]

class Flux(FluxPipeline):
    
    @torch.no_grad()
    def ODE_step(self, latents, resolution, inverse_t, inverse_text_embeddings, inverse_guidance_scale, B, K = 0, weights=None, text_embeddings=None):
        
        with torch.no_grad():
            if inverse_guidance_scale > 1.0:

                latents_noisy = latents
                latent_model_input = latents_noisy[None, :, ...].repeat(2, 1, 1, 1, 1).reshape(-1, 16, resolution[0] // 8, resolution[1] // 8, )
                tt = torch.tensor([inverse_t]).reshape(1, 1).repeat(latent_model_input.shape[0], 1).reshape(-1)
                tt = tt.to(self.device)
                # 这里的self.transformer有两个embedding
                unet_output = self.transformer(hidden_states = latent_model_input.to(self.precision_t), 
                                            timestep = tt.to(self.precision_t), 
                                            encoder_hidden_states= inverse_text_embeddings[0].to(self.precision_t),
                                            pooled_projections = inverse_text_embeddings[1].to(self.precision_t),
                                            return_dict=False,
                                                )[0].to(latent_model_input.dtype)

                unet_output = unet_output.reshape(2, -1, 16, resolution[0] // 8, resolution[1] // 8, )
                noise_pred_uncond, noise_pred_text = unet_output[:1].reshape(-1, 16, resolution[0] // 8, resolution[1] // 8, ), unet_output[1:].reshape(-1, 16, resolution[0] // 8, resolution[1] // 8, )
                delta_RFDS = noise_pred_text - noise_pred_uncond
                pred_grad = noise_pred_uncond + inverse_guidance_scale * delta_RFDS
            
            else:
                
                latents_noisy = latents
                latent_model_input = latents_noisy[None, :, ...].repeat(1, 1, 1, 1, 1).reshape(-1, 16, resolution[0] // 8, resolution[1] // 8, )
                tt = torch.tensor([inverse_t]).reshape(1, 1).repeat(latent_model_input.shape[0], 1).reshape(-1)
                tt = tt.to(self.device)

                pred_grad = self.transformer(hidden_states = latent_model_input.to(self.precision_t), 
                                            timestep = tt.to(self.precision_t), 
                                            encoder_hidden_states= inverse_text_embeddings[0][B:].to(self.precision_t),
                                            pooled_projections = inverse_text_embeddings[1][B:].to(self.precision_t),
                                            return_dict=False,
                                                )[0].to(latent_model_input.dtype)
            return pred_grad


    # Copied from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3_inpaint.StableDiffusion3InpaintPipeline._encode_vae_image
    def _encode_vae_image(self, image: torch.Tensor, generator: torch.Generator):
        if isinstance(generator, list):
            image_latents = [
                retrieve_latents(self.vae.encode(image[i : i + 1]), generator=generator[i])
                for i in range(image.shape[0])
            ]
            image_latents = torch.cat(image_latents, dim=0)
        else:
            image_latents = retrieve_latents(self.vae.encode(image), generator=generator)

        image_latents = (image_latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor

        return image_latents

    # from image to image
    def prepare_x0_latents(
        self,
        image,
        timestep,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    ):
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        height = 2 * (int(height) // self.vae_scale_factor)
        width = 2 * (int(width) // self.vae_scale_factor)

        shape = (batch_size, num_channels_latents, height, width)
        latent_image_ids = self._prepare_latent_image_ids(batch_size, height, width, device, dtype)

        if latents is not None:
            return latents.to(device=device, dtype=dtype), latent_image_ids

        image = image.to(device=device, dtype=dtype)
        image_latents = self._encode_vae_image(image=image, generator=generator)
        if batch_size > image_latents.shape[0] and batch_size % image_latents.shape[0] == 0:
            # expand init_latents for batch_size
            additional_image_per_prompt = batch_size // image_latents.shape[0]
            image_latents = torch.cat([image_latents] * additional_image_per_prompt, dim=0)
        elif batch_size > image_latents.shape[0] and batch_size % image_latents.shape[0] != 0:
            raise ValueError(
                f"Cannot duplicate `image` of batch size {image_latents.shape[0]} to {batch_size} text prompts."
            )
        else:
            image_latents = torch.cat([image_latents], dim=0)

        if timestep is not None:
            noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            latents = self.scheduler.scale_noise(image_latents, timestep, noise)
        # 这里我在encode latents以后就不在scale nosie了
        else:
            latents = image_latents

        latents = self._pack_latents(latents, batch_size, num_channels_latents, height, width)
        return latents, latent_image_ids


    # noise->x0
    @torch.no_grad()
    def flux_noise_to_x0(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 28,
        timesteps: List[int] = None,
        guidance_scale: float = 3.5,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
    ):

        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        lora_scale = (
            self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None
        )
        (
            prompt_embeds,
            pooled_prompt_embeds,
            text_ids,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )

        # 4. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels // 4
        if latents is None:
            print("now we just generate new images for you.")
        latents, latent_image_ids = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        
        num_inference_steps = len(timesteps)-1 # 我们直接根据放入的timesteps 来决定怎么loop

        # handle guidance
        if self.transformer.config.guidance_embeds:
            guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i,t in enumerate(timesteps[:-1]):
                if self.interrupt:
                    continue
                
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                t = torch.tensor([min(t,1000.0)], device=self.device) # 限制在0~1000之间
                timestep = t.expand(latents.shape[0])

                noise_pred = self.transformer(
                    hidden_states=latents,
                    timestep=timestep / 1000,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                )[0]

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype

                dt = (timesteps[i+1]-t)/1000
                dt = torch.tensor(dt).to(latents_dtype).to(device)
                print(dt)
                latents = latents+ noise_pred * dt


                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)
                    negative_pooled_prompt_embeds = callback_outputs.pop(
                        "negative_pooled_prompt_embeds", negative_pooled_prompt_embeds
                    )

                # call the callback, if provided
                # if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                progress_bar.update()

                # if XLA_AVAILABLE:
                #     xm.mark_step()

        ori_latents = latents.clone()

        latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
        latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        image = self.vae.decode(latents, return_dict=False)[0]
        image = self.image_processor.postprocess(image, output_type=output_type)
        
        return ori_latents,image[0]


    # x0->noise
    @torch.no_grad()
    def flux_x0_to_noise(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        image: PipelineImageInput = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        strength: float = 0.6,
        num_inference_steps: int = 28,
        timesteps: List[int] = None,
        guidance_scale: float = 7.0,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
    ):

        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        # 不要原因是check_inputs有一点问题
        # 1. Check inputs. Raise error if not correct
        # self.check_inputs(
        #     prompt,
        #     prompt_2,
        #     strength,
        #     height,
        #     width,
        #     prompt_embeds=prompt_embeds,
        #     pooled_prompt_embeds=pooled_prompt_embeds,
        #     callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        #     max_sequence_length=max_sequence_length,
        # )


        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False

        # 2. Preprocess image
        init_image = self.image_processor.preprocess(image, height=height, width=width)
        init_image = init_image.to(dtype=torch.float32)

        # 3. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device


        lora_scale = (
            self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None
        )
        (
            prompt_embeds,
            pooled_prompt_embeds,
            text_ids,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )

        # 4.Prepare timesteps
        if timesteps is None: # timesteps 是一个包括了起点和终点的一个timesteps
            raise ValueError("timesteps should not be None")
        # else:
        #     # print(timesteps)
        #     pass

        # 5. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels // 4

        latent_timestep = None
        if latents is not None:
            raise ValueError("latents should be None")
        else:
            # 这里的latents 
            latents, latent_image_ids = self.prepare_x0_latents(
                init_image,
                latent_timestep,
                batch_size * num_images_per_prompt,
                num_channels_latents,
                height,
                width,
                prompt_embeds.dtype,
                device,
                generator,
                latents,
            )


        num_inference_steps = len(timesteps)-1 # 我们直接根据放入的timesteps 来决定怎么loop

        # handle guidance
        if self.transformer.config.guidance_embeds:
            guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None
            

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i,t in enumerate(timesteps[:-1]):
                if self.interrupt:
                    continue
                
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                t = torch.tensor([min(t,1000.0)], device=self.device) # 限制在0~1000之间
                timestep = t.expand(latents.shape[0])

                noise_pred = self.transformer(
                    hidden_states=latents,
                    timestep=timestep / 1000,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                )[0]

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype

                dt = (timesteps[i+1]-t)/1000
                dt = torch.tensor(dt).to(latents_dtype).to(device)
                print(dt)
                latents = latents+ noise_pred * dt

                # 下面几个if 不知道是干什么的
                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)
                    negative_pooled_prompt_embeds = callback_outputs.pop(
                        "negative_pooled_prompt_embeds", negative_pooled_prompt_embeds
                    )

                # call the callback, if provided
                # if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                progress_bar.update()

                # if XLA_AVAILABLE:
                #     xm.mark_step()



        ori_latents = latents.clone()


        latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
        latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        image = self.vae.decode(latents, return_dict=False)[0]
        image = self.image_processor.postprocess(image, output_type=output_type)
          
        return ori_latents,image[0]


    # TODO fluw vae latents确定一下
    @torch.no_grad()
    def get_latents(self,batch_size=1,num_channels_latents=16,height=512,
    width=512, dtype=torch.bfloat16, device="cuda",generator=None,latents=None):
        
        height = 2 * (int(height) // self.vae_scale_factor)
        width = 2 * (int(width) // self.vae_scale_factor)

        shape = (batch_size, num_channels_latents, height, width)

        if latents is not None:
            latent_image_ids = self._prepare_latent_image_ids(batch_size, height, width, device, dtype)
            return latents.to(device=device, dtype=dtype), latent_image_ids

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        latents = self._pack_latents(latents, batch_size, num_channels_latents, height, width)

        # 暂时就不返回这个latent_image_ids, 
        # latent_image_ids = self._prepare_latent_image_ids(batch_size, height, width, device, dtype)

        # return latents, latent_image_ids
        return latents

class SD(StableDiffusionPipeline):
    
    @torch.no_grad()
    def ODE_step(self, latents, resolution, inverse_t, inverse_text_embeddings, inverse_guidance_scale, B, K = 0, weights=None, text_embeddings=None):
        
        with torch.no_grad():
            if inverse_guidance_scale > 1.0:

                latents_noisy = latents
                latent_model_input = latents_noisy[None, :, ...].repeat(2, 1, 1, 1, 1).reshape(-1, 16, resolution[0] // 8, resolution[1] // 8, )
                tt = torch.tensor([inverse_t]).reshape(1, 1).repeat(latent_model_input.shape[0], 1).reshape(-1)
                tt = tt.to(self.device)
                # 这里的self.transformer有两个embedding
                unet_output = self.transformer(hidden_states = latent_model_input.to(self.precision_t), 
                                            timestep = tt.to(self.precision_t), 
                                            encoder_hidden_states= inverse_text_embeddings[0].to(self.precision_t),
                                            pooled_projections = inverse_text_embeddings[1].to(self.precision_t),
                                            return_dict=False,
                                                )[0].to(latent_model_input.dtype)

                unet_output = unet_output.reshape(2, -1, 16, resolution[0] // 8, resolution[1] // 8, )
                noise_pred_uncond, noise_pred_text = unet_output[:1].reshape(-1, 16, resolution[0] // 8, resolution[1] // 8, ), unet_output[1:].reshape(-1, 16, resolution[0] // 8, resolution[1] // 8, )
                delta_RFDS = noise_pred_text - noise_pred_uncond
                pred_grad = noise_pred_uncond + inverse_guidance_scale * delta_RFDS
            
            else:
                
                latents_noisy = latents
                latent_model_input = latents_noisy[None, :, ...].repeat(1, 1, 1, 1, 1).reshape(-1, 16, resolution[0] // 8, resolution[1] // 8, )
                tt = torch.tensor([inverse_t]).reshape(1, 1).repeat(latent_model_input.shape[0], 1).reshape(-1)
                tt = tt.to(self.device)

                pred_grad = self.transformer(hidden_states = latent_model_input.to(self.precision_t), 
                                            timestep = tt.to(self.precision_t), 
                                            encoder_hidden_states= inverse_text_embeddings[0][B:].to(self.precision_t),
                                            pooled_projections = inverse_text_embeddings[1][B:].to(self.precision_t),
                                            return_dict=False,
                                                )[0].to(latent_model_input.dtype)
            return pred_grad

    # 每次开始必须setting一下
    def set_ddim_scheduler(self,model_key,torch_dtype):
        self.scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler", torch_dtype=torch_dtype)

    def set_ddim_inverse_scheduler(self,model_key,torch_dtype):
        self.scheduler = DDIMInverseScheduler.from_pretrained(model_key, subfolder="scheduler", torch_dtype=torch_dtype)

    # random latents, 这只是一个test,dtype还需要确认.
    def get_latents(self,batch_size=1, num_channels_latents=4, height=512, width=512, dtype=torch.float16, device="cuda", generator=None, latents=None):
        shape = (
            batch_size,
            num_channels_latents,
            int(height) // self.vae_scale_factor,
            int(width) // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents


    def prepare_x0_latents(self, image, timestep, batch_size, num_images_per_prompt, dtype, device, generator=None):
        if not isinstance(image, (torch.Tensor, Image.Image, list)):
            raise ValueError(
                f"`image` has to be of type `torch.Tensor`, `PIL.Image.Image` or list but is {type(image)}"
            )

        image = image.to(device=device, dtype=dtype)

        batch_size = batch_size * num_images_per_prompt

        if image.shape[1] == 4:
            init_latents = image

        else:
            if isinstance(generator, list) and len(generator) != batch_size:
                raise ValueError(
                    f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                    f" size of {batch_size}. Make sure the batch size matches the length of the generators."
                )

            elif isinstance(generator, list):
                if image.shape[0] < batch_size and batch_size % image.shape[0] == 0:
                    image = torch.cat([image] * (batch_size // image.shape[0]), dim=0)
                elif image.shape[0] < batch_size and batch_size % image.shape[0] != 0:
                    raise ValueError(
                        f"Cannot duplicate `image` of batch size {image.shape[0]} to effective batch_size {batch_size} "
                    )

                init_latents = [
                    retrieve_latents(self.vae.encode(image[i : i + 1]), generator=generator[i])
                    for i in range(batch_size)
                ]
                init_latents = torch.cat(init_latents, dim=0)
            else:
                init_latents = retrieve_latents(self.vae.encode(image), generator=generator)

            init_latents = self.vae.config.scaling_factor * init_latents

        if batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] == 0:
            # expand init_latents for batch_size
            deprecation_message = (
                f"You have passed {batch_size} text prompts (`prompt`), but only {init_latents.shape[0]} initial"
                " images (`image`). Initial images are now duplicating to match the number of text prompts. Note"
                " that this behavior is deprecated and will be removed in a version 1.0.0. Please make sure to update"
                " your script to pass as many initial images as text prompts to suppress this warning."
            )
            deprecate("len(prompt) != len(image)", "1.0.0", deprecation_message, standard_warn=False)
            additional_image_per_prompt = batch_size // init_latents.shape[0]
            init_latents = torch.cat([init_latents] * additional_image_per_prompt, dim=0)
        elif batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] != 0:
            raise ValueError(
                f"Cannot duplicate `image` of batch size {init_latents.shape[0]} to {batch_size} text prompts."
            )
        else:
            init_latents = torch.cat([init_latents], dim=0)

        if timestep is not None:
            shape = init_latents.shape
            noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            # get latents
            init_latents = self.scheduler.add_noise(init_latents, noise, timestep)
        latents = init_latents

        return latents



    @torch.no_grad()
    def sd_noise_to_x0(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        **kwargs,
    ):

        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        if callback is not None:
            deprecate(
                "callback",
                "1.0.0",
                "Passing `callback` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
            )
        if callback_steps is not None:
            deprecate(
                "callback_steps",
                "1.0.0",
                "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
            )

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor
        # to deal with lora scaling and other possible forward hooks

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            height,
            width,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            ip_adapter_image,
            ip_adapter_image_embeds,
            callback_on_step_end_tensor_inputs,
        )

        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # 3. Encode input prompt
        lora_scale = (
            self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
        )

        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            self.do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=lora_scale,
            clip_skip=self.clip_skip,
        )

        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
                self.do_classifier_free_guidance,
            )


        if timesteps is None:
            raise ValueError("you should put a timesteps")
        

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        if latents is None:
            print("now we just generate new images for you.")
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 6.1 Add image embeds for IP-Adapter
        added_cond_kwargs = (
            {"image_embeds": image_embeds}
            if (ip_adapter_image is not None or ip_adapter_image_embeds is not None)
            else None
        )

        # 6.2 Optionally get Guidance Scale Embedding
        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)

        # num_inference_steps = len(timesteps)-1 # 我们直接根据放入的timesteps 来决定怎么loop
        
        self.alphas_cumprod = self.scheduler.alphas_cumprod

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t) # 在ddim没有动

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    timestep_cond=timestep_cond,
                    cross_attention_kwargs=self.cross_attention_kwargs,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=self.guidance_rescale)

                # compute the previous noisy sample x_t -> x_t-1
                # latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                model_output = noise_pred
                timestep = t
                sample = latents 
                return_dict = False

                # x_t-1
                prev_timestep = timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
                
                # 2. compute alphas, betas
                alpha_prod_t = self.alphas_cumprod[timestep]
                alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod
                beta_prod_t = 1 - alpha_prod_t

                # 3. compute predicted original sample from predicted noise also called
                # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
                if self.scheduler.config.prediction_type == "epsilon":
                    pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
                    pred_epsilon = model_output
                elif self.scheduler.config.prediction_type == "sample":
                    pred_original_sample = model_output
                    pred_epsilon = (sample - alpha_prod_t ** (0.5) * pred_original_sample) / beta_prod_t ** (0.5)
                elif self.scheduler.config.prediction_type == "v_prediction":
                    pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
                    pred_epsilon = (alpha_prod_t**0.5) * model_output + (beta_prod_t**0.5) * sample
                else:
                    raise ValueError(
                        f"prediction_type given as {self.scheduler.config.prediction_type} must be one of `epsilon`, `sample`, or"
                        " `v_prediction`"
                    )

                # 4. Clip or threshold "predicted x_0"
                if self.scheduler.config.thresholding:
                    pred_original_sample = self.scheduler._threshold_sample(pred_original_sample)
                elif self.scheduler.config.clip_sample:
                    pred_original_sample = pred_original_sample.clamp(
                        -self.scheduler.config.clip_sample_range, self.scheduler.config.clip_sample_range
                    )

                # 一般eta是0.0
                # 5. compute variance: "sigma_t(η)" -> see formula (16)
                # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
                # variance = self._get_variance(timestep, prev_timestep)
                # std_dev_t = eta * variance ** (0.5)

                # if use_clipped_model_output:
                #     # the pred_epsilon is always re-derived from the clipped x_0 in Glide
                #     pred_epsilon = (sample - alpha_prod_t ** (0.5) * pred_original_sample) / beta_prod_t ** (0.5)

                # 6. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
                # pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (0.5) * pred_epsilon
                pred_sample_direction = (1 - alpha_prod_t_prev) ** (0.5) * pred_epsilon


                # 7. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
                # prev_sample = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction
                latents = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction


                # TODO: check 下面的函数
                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                # call the callback, if provided
                num_warmup_steps = 0
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

                # if XLA_AVAILABLE:
                #     xm.mark_step()

        ori_latents = latents.clone()

        image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False, generator=generator)[
            0
        ]
        # image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)

        # if has_nsfw_concept is None:
        #     do_denormalize = [True] * image.shape[0]
        # else:
        #     do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]
        # image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        image = self.image_processor.postprocess(image, output_type=output_type)

        return latents, image[0]


    @torch.no_grad()
    def sd_x0_to_noise(
        self,
        prompt: Union[str, List[str]] = None,
        image: PipelineImageInput = None,
        strength: float = 0.8, # 这个参数我们不用使用
        num_inference_steps: Optional[int] = 50,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        guidance_scale: Optional[float] = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: Optional[float] = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        clip_skip: int = None,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        **kwargs,
    ):

        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        if callback is not None:
            deprecate(
                "callback",
                "1.0.0",
                "Passing `callback` as an input argument to `__call__` is deprecated, consider use `callback_on_step_end`",
            )
        if callback_steps is not None:
            deprecate(
                "callback_steps",
                "1.0.0",
                "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider use `callback_on_step_end`",
            )

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        self._guidance_scale = guidance_scale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        
        # 3. Encode input prompt
        text_encoder_lora_scale = (
            self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
        )
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            self.do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
            clip_skip=self.clip_skip,
        )
        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
                self.do_classifier_free_guidance,
            )

        # 4. Preprocess image
        image = self.image_processor.preprocess(image)

        # we put timesteps
        
        latent_timestep = None
        # 6. Prepare latent variables
        latents = self.prepare_x0_latents(
            image=image,
            timestep=latent_timestep,
            batch_size=batch_size,
            num_images_per_prompt=num_images_per_prompt,
            dtype=prompt_embeds.dtype,
            device=device,
            generator=generator,
        )

        # 7. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7.1 Add image embeds for IP-Adapter
        added_cond_kwargs = (
            {"image_embeds": image_embeds}
            if ip_adapter_image is not None or ip_adapter_image_embeds is not None
            else None
        )

        # 7.2 Optionally get Guidance Scale Embedding
        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)

        self.alphas_cumprod = self.scheduler.alphas_cumprod

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    timestep_cond=timestep_cond,
                    cross_attention_kwargs=self.cross_attention_kwargs,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]

                
                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)


                model_output = noise_pred
                timestep = t
                sample = latents
                return_dict = False

                # 1. get previous step value (=t+1)
                prev_timestep = timestep
                timestep = min(
                    timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, self.scheduler.config.num_train_timesteps - 1
                )

                # 2. compute alphas, betas
                # change original implementation to exactly match noise levels for analogous forward process
                alpha_prod_t = self.alphas_cumprod[timestep] if timestep >= 0 else self.scheduler.initial_alpha_cumprod
                alpha_prod_t_prev = self.alphas_cumprod[prev_timestep]

                beta_prod_t = 1 - alpha_prod_t

                # 3. compute predicted original sample from predicted noise also called
                # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
                if self.scheduler.config.prediction_type == "epsilon":
                    pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
                    pred_epsilon = model_output
                elif self.scheduler.config.prediction_type == "sample":
                    pred_original_sample = model_output
                    pred_epsilon = (sample - alpha_prod_t ** (0.5) * pred_original_sample) / beta_prod_t ** (0.5)
                elif self.scheduler.config.prediction_type == "v_prediction":
                    pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
                    pred_epsilon = (alpha_prod_t**0.5) * model_output + (beta_prod_t**0.5) * sample
                else:
                    raise ValueError(
                        f"prediction_type given as {self.scheduler.config.prediction_type} must be one of `epsilon`, `sample`, or"
                        " `v_prediction`"
                    )

                # 4. Clip or threshold "predicted x_0"
                if self.scheduler.config.clip_sample:
                    pred_original_sample = pred_original_sample.clamp(
                        -self.scheduler.config.clip_sample_range, self.scheduler.config.clip_sample_range
                    )

                # 5. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
                pred_sample_direction = (1 - alpha_prod_t_prev) ** (0.5) * pred_epsilon

                # 6. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
                latents = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction

                # TODO: check 下面的函数
                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                # call the callback, if provided
                num_warmup_steps = 0
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

                # if XLA_AVAILABLE:
                #     xm.mark_step()

                

        # if not output_type == "latent":
        #     image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False, generator=generator)[
        #         0
        #     ]
        #     image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
        # else:
        #     image = latents
        #     has_nsfw_concept = None

        ori_latents = latents.clone()

        image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False, generator=generator)[
            0
        ]
        # image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)

        # if has_nsfw_concept is None:
        #     do_denormalize = [True] * image.shape[0]
        # else:
        #     do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]
        # image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        image = self.image_processor.postprocess(image, output_type=output_type)

        return latents, image


# 就是这个timesteps应该包括起点和终点0-1000
def get_custom_timesteps(num_infrence_steps):
    
    dt = 1/num_infrence_steps
    timesteps = [0.0]
    current_t = 0
    for i in range(num_infrence_steps-1):
        current_t = current_t+dt*1000
        timesteps.append(current_t)
    timesteps.append(1000.0)
    timesteps = torch.tensor(timesteps, device="cuda")
    return timesteps

# 将timesteps flow 和 diffusion的，我独立写出来

# 4. Prepare timesteps
# sigmas is None
# timesteps, num_inference_steps = retrieve_timesteps(
#     self.scheduler, num_inference_steps, device, timesteps, sigmas
# )

# 这里的self指的是schedule，ddim
def set_ddim_timesteps(self,num_inference_steps: int, device: Union[str, torch.device] = "cuda",
set_alpha_to_one=False):
        
        self.num_inference_steps = num_inference_steps
        self.initial_alpha_cumprod = torch.tensor(1.0) if set_alpha_to_one else self.alphas_cumprod[0]

        # "linspace", "leading", "trailing" corresponds to annotation of Table 2. of https://arxiv.org/abs/2305.08891
        if self.config.timestep_spacing == "linspace":
            timesteps = (
                np.linspace(0, self.config.num_train_timesteps - 1, num_inference_steps)
                .round()[::-1]
                .copy()
                .astype(np.int64)
            )
        elif self.config.timestep_spacing == "leading":
            step_ratio = self.config.num_train_timesteps // num_inference_steps
            # creates integer timesteps by multiplying by ratio
            # casting to int to avoid issues when num_inference_step is power of 3
            timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
            timesteps += self.config.steps_offset
        elif self.config.timestep_spacing == "trailing":
            step_ratio = self.config.num_train_timesteps / num_inference_steps
            # creates integer timesteps by multiplying by ratio
            # casting to int to avoid issues when num_inference_step is power of 3
            timesteps = np.round(np.arange(self.config.num_train_timesteps, 0, -step_ratio)).astype(np.int64)
            timesteps -= 1
        else:
            raise ValueError(
                f"{self.config.timestep_spacing} is not supported. Please make sure to choose one of 'leading' or 'trailing'."
            )

        timesteps = torch.from_numpy(timesteps).to(device)
        return timesteps

# 这里的self指的是schedule，flow
def set_flow_timesteps(
    self,
    num_inference_steps: int = None,
    device: Union[str, torch.device] = "cuda",
    sigmas: Optional[List[float]] = None,
    mu: Optional[float] = None,
):
    """
    Sets the discrete timesteps used for the diffusion chain (to be run before inference).

    Args:
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
    """
    # 暂时支持use_dynamic_shifting
    self.config.use_dynamic_shifting = False

    if self.config.use_dynamic_shifting and mu is None:
        raise ValueError(" you have a pass a value for `mu` when `use_dynamic_shifting` is set to be `True`")

    if sigmas is None:
        # self.num_inference_steps = num_inference_steps
        timesteps = np.linspace(
            self._sigma_to_t(self.sigma_max), self._sigma_to_t(self.sigma_min), num_inference_steps
        )

        sigmas = timesteps / self.config.num_train_timesteps

    if self.config.use_dynamic_shifting:
        sigmas = self.time_shift(mu, 1.0, sigmas)
    else:
        sigmas = self.config.shift * sigmas / (1 + (self.config.shift - 1) * sigmas)

    sigmas = torch.from_numpy(sigmas).to(dtype=torch.float32, device=device)
    timesteps = sigmas * self.config.num_train_timesteps


    timesteps = timesteps.to(device=device)
    timesteps = torch.cat([timesteps,torch.zeros(1, device=sigmas.device)])
    # timesteps = torch.cat([timesteps,torch.tensor(1.0000, device=sigmas.device).view(1)])

    self.sigmas = torch.cat([sigmas, torch.zeros(1, device=sigmas.device)])

    return timesteps
    # self._step_index = None
    # self._begin_index = None





# if __name__=="__main__":

    ################################ SD3 ################################
    # # # TODO:scheduler写出去
    # pipe = SD3.from_pretrained("/mnt/workspace/common/models/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16)
    # pipe = pipe.to("cuda")

    # # image = pipe(
    # #     "A hamburger",
    # #     negative_prompt="",
    # #     num_inference_steps=28,
    # #     guidance_scale=7.0,
    # # ).images[0]

    # # print(image.dtype)
    # # image.save("images/output.png")


    # # ###############     ###############
    # image_path = "images/output_.png"
    # image = Image.open(image_path)

    # num_infrence_steps = 50
    # timesteps = get_custom_timesteps(num_infrence_steps)
    # # timesteps = timesteps[]
    # print(timesteps)

    # latents,image = pipe.sd3_x0_to_noise(
    # "A cat holding a sign that says hello world",
    # negative_prompt="",
    # # num_inference_steps=28,
    # timesteps = timesteps,
    # image = image,
    # guidance_scale=1.0,
    # )
    # image.save(f"images/sd3_x0_to_noise_{num_infrence_steps}_start0.png")

    # timesteps =timesteps[::-1]
    # print(timesteps)

    # latents,image = pipe.sd3_noise_to_x0(
    # "A cat holding a sign that says hello world",
    # negative_prompt="",
    # # num_inference_steps=28,
    # timesteps = timesteps,
    # # image = image,
    # latents=latents,
    # # latents=None,
    # guidance_scale=1.0,
    # )
    # # image.save(f"images/sd3_noise_to_x0_{num_infrence_steps}.png")
    # image.save(f"images/sd3_noise_to_x0_{num_infrence_steps}_start0.png")

    # # ###############     ###############

    # # latents,image = pipe.get_latents()
    # # image.save("images_1/noise.png")
    # # num_infrence_steps = 20
    # # timesteps = get_custom_timesteps(num_infrence_steps)
    # # timesteps_inverse = timesteps[::-1]

    # # latents,image = pipe.sd3_noise_to_x0(
    # # "A cat holding a sign that says hello world",
    # # negative_prompt="",
    # # # num_inference_steps=28,
    # # timesteps = timesteps_inverse,
    # # # image = image,
    # # latents=latents,
    # # guidance_scale=7.0,
    # # )
    # # image.save(f"images_1/sd3_noise_to_x0_{num_infrence_steps}.png")

    # # latents,image = pipe.sd3_x0_to_noise(
    # # "A cat holding a sign that says hello world",
    # # negative_prompt="",
    # # # num_inference_steps=28,
    # # timesteps = timesteps,
    # # image = image,
    # # guidance_scale=1.0,
    # # )
    # # image.save(f"images_1/sd3_x0_to_noise_{num_infrence_steps}.png")



    # # pipe = Flux.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
    # # pipe.enable_model_cpu_offload()


    # # prompt = "a tiny astronaut hatching from an egg on the moon"
    # # out = pipe(
    # #     prompt=prompt,
    # #     guidance_scale=3.5,
    # #     height=768,
    # #     width=1360,
    # #     num_inference_steps=50,
    # # ).images[0]
    # # out.save("image.png")
    ################################ SD3 ################################

    ################################ Flux ################################

    # pipe = Flux.from_pretrained("/mnt/workspace/common/models/FLUX.1-dev", torch_dtype=torch.bfloat16)
    # pipe.enable_model_cpu_offload()


    # latents = pipe.get_latents()
    # num_inference_steps=50
    # # timesteps=get_custom_timesteps(50)
    # timesteps = set_flow_timesteps(self = pipe.scheduler, num_inference_steps = num_inference_steps)
    # # sigmas

    # print(timesteps)


    # image_path = "images/output_.png"
    # image = Image.open(image_path)


    # latents,image = pipe.flux_x0_to_noise(
    # prompt= "A cat holding a sign that says hello world",
    # # num_inference_steps=28,
    # timesteps = timesteps,
    # image = image,
    # guidance_scale=1.0,
    # )
    # image.save(f"images_flux/flux_x0_to_noise_{num_infrence_steps}.png")


    # timesteps = timesteps[::-1]
    # print(timesteps)
    # latents,image = pipe.flux_noise_to_x0(
    # "A cat holding a sign that says hello world",
    # # negative_prompt="",
    # # num_inference_steps=28,
    # timesteps = timesteps,
    # # image = image,
    # latents=latents,
    # # latents=None,
    # guidance_scale=1.0,
    # )

    # image.save(f"images_flux/flux_noise_to_x0_{num_infrence_steps}_start0.png")
    ################################ Flux ################################


    ################################ SD ################################

    # image_path = "images_sd/sd_noise_to_x0_50.png"
    # image = Image.open(image_path)

    # pipe = SD.from_pretrained("/mnt/workspace/common/models/stable-diffusion-v1-5", torch_dtype=torch.float16)
    # # pipe.set_ddim_scheduler("/mnt/workspace/common/models/stable-diffusion-v1-5", torch_dtype=torch.float16)
    # pipe.set_ddim_scheduler("/mnt/workspace/common/models/stable-diffusion-v1-5", torch_dtype=torch.float16)
    # pipe = pipe.to("cuda")
    # prompt = "a photo of an astronaut riding a horse on mars"
    
    # num_inference_steps = 50
    # timesteps = set_ddim_timesteps(pipe.scheduler,num_inference_steps=num_inference_steps)

    # timesteps = torch.flip(timesteps,dims=[0])
    # print(timesteps)

    # latents,image = pipe.sd_x0_to_noise(
    #     prompt=prompt,
    #     image=image,
    #     num_inference_steps=num_inference_steps,
    #     timesteps=timesteps,
    #     guidance_scale=1.0
    #     )
    
    # # num_inference_steps=50
    # image[0].save(f"images_sd/sd_x0_to_noise_{num_inference_steps}_.png")

    # timesteps = torch.flip(timesteps,dims=[0])
    # print(timesteps)
    # # latents = pipe.get_latents()
    # latents,image = pipe.sd_noise_to_x0(
    #     prompt=prompt,
    #     latents=latents,
    #     num_inference_steps=num_inference_steps,
    #     timesteps=timesteps,
    #     guidance_scale=1.0
    #     )
    # image[0].save(f"images_sd/sd_noise_to_x0_{num_inference_steps}_.png")

