import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import SanaTransformer2DModel, AutoencoderDC, DPMSolverMultistepScheduler
import torch.nn.functional as F


class Diffloss(nn.Module):
    def __init__(self, diffusion_config):
        """
        Args:
            diffusion_config (dict): 配置字典，应包含以下键：
                - 'model_name': 用于加载扩散模型（SanaTransformer2DModel）的路径或名称
                - 'vae_model_name': 用于加载 VAE 模型的路径或名称
                - 'scheduler_model_name': 用于加载 scheduler 的路径或名称
        """
        super(Diffloss, self).__init__()
        # 加载扩散网络（例如 SanaTransformer2DModel）
        self.model = SanaTransformer2DModel.from_pretrained(diffusion_config['model_name'])
        # 加载 VAE 模型，用于将图像编码到 latent 空间
        self.vae = AutoencoderDC.from_pretrained(diffusion_config['vae_model_name'])
        # 加载 scheduler，该组件封装了向 latent 添加噪声的操作及时间步信息
        self.scheduler = DPMSolverMultistepScheduler.from_pretrained(diffusion_config['scheduler_model_name'])
        # 从 scheduler 配置中获取训练时的总时间步数（如果配置中没有该项，可设定一个默认值，如 1000）
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
        # 编码后一般返回一个包含 latent_dist 的对象，从中采样得到 latent
        vae_output = self.vae.encode(clean_image)
        latents = vae_output.latent_dist.sample()  # [bs, latent_channels, H', W']

        # -------------------------------
        # 3. 为每个样本采样随机噪声和时间步
        noise = torch.randn_like(latents, device=latents.device)
        bs = latents.shape[0]
        # 随机采样时间步，从 0 到 num_train_timesteps - 1 之间
        timesteps = torch.randint(0, self.num_train_timesteps, (bs,), device=latents.device).long()

        # -------------------------------
        # 4. 向 latent 添加噪声，scheduler 内部封装了对应的算法
        # 传入原始 latent、噪声和时间步，返回加噪后的 latent
        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)

        # -------------------------------
        # 5. 扩散模型预测噪声
        # 注意：此处参数名称需要与扩散模型的 forward 方法保持一致。
        # 如果你的扩散模型支持文本条件，则传入 prompt_embeds 和 prompt_attention_mask
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