# --------------------------------------------------------
# Mini1o dit
# Copyright (c) 2025 Yilin Jia
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import inspect
from diffusers import SanaTransformer2DModel, AutoencoderDC, DPMSolverMultistepScheduler
from .config_dit import DitConfig
from typing import Optional, Dict, Any, List, Tuple, Union

def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


class Mini1oDiT(nn.Module):
    def __init__(self, 
                 config: DitConfig, 
                 pretrained_path: Optional[str] = None, 
                 **kwargs):
        """
        Args:
            diffusion_config (dict): 配置字典，应包含以下键：
                - 'model_name': 用于加载扩散模型（SanaTransformer2DModel）的路径或名称
                - 'vae_model_name': 用于加载 VAE 模型的路径或名称
                - 'scheduler_model_name': 用于加载 scheduler 的路径或名称
        """
        super(Mini1oDiT, self).__init__()
        if pretrained_path is not None:
            self.model = SanaTransformer2DModel.from_pretrained(pretrained_path, subfolder = 'transformers')
            self.vae = AutoencoderDC.from_pretrained(pretrained_path, subfolder = 'vae')
            self.scheduler =  DPMSolverMultistepScheduler.from_pretrained(pretrained_path, subfolder = 'scheduler')
        else:
            self.model = SanaTransformer2DModel(**config.model_config.to_dict())
            self.vae = AutoencoderDC(**config.vae_config.to_dict())
            self.scheduler = DPMSolverMultistepScheduler(**config.scheduler_config.to_dict())
        self.num_train_timesteps = getattr(self.scheduler.config, 'num_train_timesteps', 1000)

    def forward(self, 
                clean_image,           # 原始图像，形状为 [bs, 3, H, W]
                prompt_embeds=None,    # 可选，文本编码向量，形状为 [bs, seq_len, hidden_dim]
                prompt_attention_mask=None,  # 可选，文本注意力 mask，形状为 [bs, seq_len]
                **kwargs):
        """
        训练过程中计算扩散损失。
        
        1. 利用 VAE 对原图进行编码，获得 latent 表示；
        2. 将 latent 乘以 sigma_data 进行尺度调整（匹配训练时噪声尺度）；
        3. 随机采样噪声和时间步，并用 scheduler.add_noise 将噪声添加到 latent 上；
        4. 将添加噪声后的 latent 和时间步传入扩散模型预测噪声；
        5. 计算预测噪声与实际添加噪声之间的均方误差损失。
        """
        # -------------------------------
        # 1. VAE 编码
        vae_output = self.vae.encode(clean_image)
        if hasattr(vae_output, "latent_dist"):
            latents = vae_output.latent_dist.sample()
        else:
            latents = vae_output.latent
        # -------------------------------
        # 3. 为每个样本采样随机噪声和时间步
        noise = torch.randn_like(latents, device=latents.device)
        bs = latents.shape[0]
        # 随机采样时间步，从 0 到 num_train_timesteps - 1 之间
        timesteps = torch.randint(0, self.num_train_timesteps, (bs,), device=latents.device).long()

        # -------------------------------
        # 4. 向 latent 添加噪声，scheduler 内部封装了对应的算法
        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)

        # -------------------------------
        # 5. 扩散模型预测噪声
        model_output = self.model(
            hidden_states=noisy_latents,
            timestep=timesteps,
            encoder_hidden_states=prompt_embeds,
            encoder_attention_mask=prompt_attention_mask,
            **kwargs
        ).sample

        # -------------------------------
        # 6. 计算预测噪声与真实噪声之间的均方误差损失
        loss = F.mse_loss(model_output, noise)
        return loss

    def sample(self,
               prompt_embeds: torch.Tensor = None,
               prompt_attention_mask: torch.Tensor = None,
               negative_prompt_embeds: torch.Tensor = None,
               negative_prompt_attention_mask: torch.Tensor = None,
               num_inference_steps: int = 50,
               guidance_scale: float = 1.0,
               batch_size: int = 1,
               height: int = 256,
               width: int = 256,
               generator: torch.Generator = None,
               latents: torch.Tensor = None,
               eta: float = 0.0,
               **kwargs) -> torch.Tensor:
        """
        采样函数

        Args:
            prompt_embeds (torch.Tensor, optional): 文本编码向量，形状为 [bs, seq_len, hidden_dim]。若使用 classifier-free guidance，
                则要求同时提供 negative_prompt_embeds。
            prompt_attention_mask (torch.Tensor, optional): 对应的注意力 mask。
            negative_prompt_embeds (torch.Tensor, optional): 无条件文本编码，用于 classifier-free guidance。
            negative_prompt_attention_mask (torch.Tensor, optional): 无条件的注意力 mask。
            num_inference_steps (int): 去噪采样步数。
            guidance_scale (float): classifier-free guidance 的权重，当 > 1 时启用。默认为 1.0。
            batch_size (int): 样本数量。
            height (int): 生成图像的高度（像素）。
            width (int): 生成图像的宽度（像素）。
            generator (torch.Generator, optional): 用于随机数生成的生成器。
            latents (torch.Tensor, optional): 初始的 latent 张量；若为 None，则采样随机噪声。
            eta (float): DDIM 采样中使用的 eta 参数。
            **kwargs: 传递给扩散模型的其他参数。

        Returns:
            torch.Tensor: 解码后的生成图像张量。
        """
        # 获取设备（假设 self.model、self.vae 均在同一设备上）
        device = next(self.model.parameters()).device

        # 计算 VAE 的缩放因子，参照 SanaPipeline 的计算逻辑
        if hasattr(self.vae.config, "encoder_block_out_channels"):
            vae_scale_factor = 2 ** (len(self.vae.config.encoder_block_out_channels) - 1)
        else:
            vae_scale_factor = 32

        # 计算 latent 的形状
        latent_channels = self.model.config.in_channels
        latent_height = height // vae_scale_factor
        latent_width = width // vae_scale_factor

        # 若没有传入初始 latent，则采样随机噪声
        if latents is None:
            latents = torch.randn(
                (batch_size, latent_channels, latent_height, latent_width), device=device
            ).to(torch.bfloat16)
        else:
            latents = latents.to(device).to(torch.bfloat16)
        # 使用 retrieve_timesteps 获取采样时间步（注意 timesteps 顺序一般为降序）
        timesteps, _ = retrieve_timesteps(self.scheduler, num_inference_steps, device)

        # 准备 scheduler.step 的额外参数
        extra_step_kwargs = {}
        step_params = inspect.signature(self.scheduler.step).parameters
        if "eta" in step_params:
            extra_step_kwargs["eta"] = eta
        if "generator" in step_params:
            extra_step_kwargs["generator"] = generator

        # 采样去噪循环
        for t in timesteps:
            # 若使用 classifier-free guidance，则需要复制 latent，并拼接条件与无条件的文本编码
            if guidance_scale > 1.0 and (prompt_embeds is not None and negative_prompt_embeds is not None):
                latent_input = torch.cat([latents, latents], dim=0)
                encoder_hidden_states = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
                if prompt_attention_mask is not None and negative_prompt_attention_mask is not None:
                    encoder_attention_mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask], dim=0)
                else:
                    encoder_attention_mask = None
            else:
                latent_input = latents
                encoder_hidden_states = prompt_embeds
                encoder_attention_mask = prompt_attention_mask

            # 将当前时间步扩展到与 latent_input 相同的 batch 大小
            t_batch = t.expand(latent_input.shape[0]).to(latents.dtype)

            # 利用扩散模型预测噪声
            noise_pred = self.model(
                hidden_states=latent_input,
                timestep=t_batch,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                **kwargs
            ).sample
            # noise_pred = noise_pred.float()

            # 若使用 classifier-free guidance，对预测噪声进行融合
            if guidance_scale > 1.0 and (prompt_embeds is not None and negative_prompt_embeds is not None):
                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2, dim=0)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

            # 使用 scheduler.step 更新 latent
            latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

        # 解码 latent 得到图像
        latents = latents.to(self.vae.dtype)
        # 注意：VAE 解码前需要除以配置中的缩放因子
        latents_dec = latents / self.vae.config.scaling_factor
        images = self.vae.decode(latents_dec, return_dict=False)[0]

        return images
    
    def from_pretrained(self, pretrained_model_name_or_path: str, **kwargs):
        """
        从预训练模型加载参数，并初始化 Mini1o 模型。

        参数：
            pretrained_model_name_or_path (str): 预训练模型的路径或名称。
            **kwargs: 其他传入参数。
        """
        # 加载预训练模型的配置
        config = DitConfig.from_pretrained(pretrained_model_name_or_path)
        model = self.__class__(config,  pretrained_model_name_or_path, **kwargs)
        return model
