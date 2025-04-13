import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from diffusers import UNet2DConditionModel, AutoencoderKL, DDPMScheduler
from torchvision import transforms
from PIL import Image
import numpy as np
from typing import Optional, Dict, Any
try:
    from diffloss import Diffloss
except:
    from .diffloss import Diffloss

class Connector(nn.Module):
    def __init__(self, hidden_dim, diffusion_dim, num_layers=4, nhead=8):
        """
        Connector 模块先用 Transformer Encoder 对输入进行对齐，
        再通过 Linear Projection 映射到 diffusion 模型的条件嵌入空间（例如 768 维）。
        """
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.proj = nn.Linear(hidden_dim, diffusion_dim)

    def forward(self, x):
        x = self.encoder(x)
        return self.proj(x)
    
class Mini4o:
    def __init__(self, mllm_config, diffusion_config):
        self.mllm = AutoModelForCausalLM.from_pretrained(mllm_config['model_name'])
        self.connector = Connector(hidden_dim=mllm_config['hidden_dim'], diffusion_dim=diffusion_config['diffusion_dim'])
        self.num_queries = self.mllm_config.num_image_gen_tokens
        self.hidden_dim = self.mllm.config.hidden_size
        self.meta_queries = nn.Parameter(torch.randn(self.num_queries, self.hidden_dim))
        self.diffloss = Diffloss(diffusion_config)

    def forward(
        self,
        *args,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        gen_pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        image_gen_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ):
        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)
            if pixel_values is not None:
                pixel_values = pixel_values.type(self.visual.dtype)
                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
                n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
                n_image_features = image_embeds.shape[0]
                if n_image_tokens != n_image_features:
                    raise ValueError(
                        f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                    )

                mask = input_ids == self.config.image_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                image_mask = mask_expanded.to(inputs_embeds.device)

                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
                video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
                n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
                n_video_features = video_embeds.shape[0]
                if n_video_tokens != n_video_features:
                    raise ValueError(
                        f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                    )

                mask = input_ids == self.config.video_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                video_mask = mask_expanded.to(inputs_embeds.device)

                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

            # Inject meta queries at image_gen_pad_token_id positions
            if gen_pixel_values is not None and input_ids is not None:
                image_gen_pad_token_id = self.mllm.config.image_gen_pad_token_id
                mask = input_ids == image_gen_pad_token_id
                total_tokens = mask.sum().item()
                total_features = image_gen_grid_thw.shape[0] * self.num_queries

                if total_tokens != total_features:
                    raise ValueError(
                        f"Image-gen features and tokens do not match: tokens: {total_tokens}, features {total_features}"
                    )

                B, L = input_ids.shape
                H = self.mllm.config.hidden_size
                meta_queries = self.meta_queries.unsqueeze(0).expand(image_gen_grid_thw.shape[0], -1, -1)
                meta_queries = meta_queries.reshape(-1, H)

                inputs_embeds_flat = inputs_embeds.view(-1, H)
                inputs_embeds_flat[mask.view(-1)] = meta_queries.to(inputs_embeds.dtype)
                inputs_embeds = inputs_embeds_flat.view(B, L, H)

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        mllm_output = self.mllm.forward(
            *args,
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

        # 把 mask的地方，由meta query提取出来的特征拿出来去connector， 然后再送去diffusion
        hidden_states = mllm_output.last_hidden_state  # shape: [B, L, H]
        image_gen_pad_token_id = self.mllm.config.image_gen_pad_token_id
        meta_mask = input_ids == image_gen_pad_token_id  # bool 型掩码

        # 提取出所有 meta query 位置的特征，结果形状为 (num_meta_tokens, H)
        meta_features = hidden_states[meta_mask]

        # 若提供了 image_gen_grid_thw，则将提取的特征重新 reshape 成网格形式，方便后续条件生成
        # 变成 n * 256 * H
        meta_features = meta_features.view(-1, self.num_queries, hidden_states.shape[-1])

        # 通过 Connector 模块将 meta_features 映射到 diffusion 模型需要的条件嵌入空间
        diffusion_condition = self.connector(meta_features)  # shape: (num_images, tokens_per_image, diffusion_dim)

        diff_loss = self.diffloss(
            clean_image=gen_pixel_values,
            prompt_embeds=diffusion_condition,
            image_grid_thw=image_gen_grid_thw,
            **kwargs
        )
        return mllm_output, diff_loss
        # return mllm_output, diffusion_output, decoded_images