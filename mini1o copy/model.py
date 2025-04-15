# --------------------------------------------------------
# Mini1o
# Copyright (c) 2025 Yilin Jia
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import torch
import torch.nn as nn

try:
    from dit import Mini1oDiT
    from mllm import Mini1oMLLM
except:
    from .dit import Mini1oDiT
    from .mllm import Mini1oMLLM


# Connector 模块：用于将输入（例如 image embedding）转换并对齐
class Mini1oConnector(nn.Module):
    def __init__(self, hidden_dim, diffusion_dim=2304, num_layers=6, nhead=8):
        """
        先使用 Transformer Encoder 对输入进行对齐，再通过 Linear Projection 映射到目标维度
        """
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.proj = nn.Linear(hidden_dim, diffusion_dim)

    def forward(self, x):
        x = self.encoder(x)
        return self.proj(x)

class Mini1o(nn.Module):
    def __init__(self, dit_config: dict, mllm_config: dict, connector_config: dict):
        """
        初始化组合模型 Mini1o。

        参数：
            dit_config (dict): 用于初始化 Mini1oDiT 的配置字典。
            mllm_config (dict): 用于初始化 Mini1oMLLM 的配置字典，其中需要包含 key "num_img_gen_tokens"。
            connector_config (dict): Connector 参数字典，可包含如下键：
                - diffusion_dim: 扩散模型条件嵌入的维度（默认2304）
                - num_layers: TransformerEncoder 层数（默认6）
                - nhead: 注意力头个数（默认8）
        """
        super(Mini1o, self).__init__()
        self.num_img_gen_tokens = mllm_config.get('num_img_gen_tokens', 64)

        self.dit = Mini1oDiT(dit_config)
        self.mllm = Mini1oMLLM(mllm_config)

        ## connector
        self.hidden_dim = self.mllm.hidden_size
        diffusion_dim = connector_config.get('diffusion_dim', 2304)
        num_layers = connector_config.get('num_layers', 6)
        nhead = connector_config.get('nhead', 8)
        self.connector = Mini1oConnector(self.hidden_dim, diffusion_dim, num_layers, nhead)
    def forward(self, pixel_values: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor, clean_image: torch.Tensor, **kwargs):
        """
        前向计算（训练模式）：
        
        1. 通过 mllm 得到文本（及图像）对应的隐藏状态及条件 mask；
        2. 利用条件 mask 从 mllm 的最后一层隐藏状态中抽取图像生成 token；
        3. 通过 connector 对这些 token 进行对齐，得到扩散模型所需的条件嵌入；
        4. 将条件嵌入传入 dit，计算扩散模型损失（例如均方误差）。

        参数：
            pixel_values: 输入图像（如用于 ViT 特征提取），形状 [B, C, H, W]。
            input_ids: 输入的 token ids，形状 [B, seq_len]。
            attention_mask: 注意力 mask，形状 [B, seq_len]。
            clean_image: 供 dit 计算损失使用的原始图像，形状 [B, 3, H, W]。
            **kwargs: 其他传入 mllm 或 dit 的参数。

        返回：
            loss: 扩散模型计算得到的训练损失。
        """
        # 1. 通过 mllm 得到输出（包括 hidden_states 与图像生成 token 的 mask）
        mllm_out = self.mllm(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
        # 2. 从 mllm 的最后一层隐藏状态中提取 image generation token 对应的条件表示
        # hidden_states 的形状 [B, seq_len, hidden_dim]
        hidden_state = mllm_out.hidden_states[-1]
        # mllm forward 返回的 condition_mask（通常是通过 input_ids 得到的布尔 mask）
        condition_mask = mllm_out.condition_mask  # 形状 [B, seq_len]，True 表示图像生成 token
        condition_tokens= hidden_state.reshape(-1, self.hidden_dim)[condition_mask].reshape(-1,self.num_img_gen_tokens, self.hidden_dim)
        # 3. 通过 connector 将 mllm 得到的条件信息映射到扩散模型的条件嵌入空间
        conditioned = self.connector(condition_tokens)  # 形状 [B, num_img_gen_tokens, diffusion_dim]
        
        # 4. 将条件信息传给 dit（扩散模型部分），计算训练损失
        loss = self.dit(
            clean_image=clean_image, # [n * 3 * H * W]
            prompt_embeds=conditioned,
            **kwargs
        )
        return loss

    @torch.no_grad()
    def generate(self, pixel_values: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor, generator: torch.Generator = None, num_inference_steps: int = 50, guidance_scale: float = 1.0, **kwargs) -> torch.Tensor:
        """
        推理时生成图像：
        
        1. 同 forward，先调用 mllm 得到条件 token，并通过 connector 映射；
        2. 将条件嵌入传给 dit 的 sample 方法，经过扩散采样后解码生成图像。

        参数：
            pixel_values: 输入的图像（例如用于 ViT 特征抽取），形状 [B, C, H, W]。
            input_ids: 输入的 token ids，形状 [B, seq_len]。
            attention_mask: 注意力 mask。
            generator: 用于采样的随机数生成器。
            num_inference_steps: 扩散模型采样时的步数。
            guidance_scale: classifier-free guidance 权重。
            **kwargs: 其他传入 mllm 或 dit 的参数。

        返回：
            images: 生成的图像张量。
        """
        # 1. 通过 mllm 得到 hidden_states 和 condition mask
        mllm_out = self.mllm(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
        hidden_state = mllm_out.hidden_states[-1]
        condition_mask = mllm_out.condition_mask  # [B, seq_len]
        batch_size = hidden_state.shape[0]
        condition_tokens = []
        for i in range(batch_size):
            tokens = hidden_state[i][condition_mask[i]]
            if tokens.shape[0] != self.num_img_gen_tokens:
                raise ValueError(
                    f"期望每个样本有 {self.num_img_gen_tokens} 个图像生成 token，但第 {i} 个样本仅有 {tokens.shape[0]} 个"
                )
            condition_tokens.append(tokens)
        condition_tokens = torch.stack(condition_tokens, dim=0)
        
        # 2. 通过 connector 映射到扩散模型条件嵌入空间
        conditioned = self.connector(condition_tokens)  # [B, num_img_gen_tokens, diffusion_dim]
        
        # 3. 调用 dit 的 sample 方法生成图像
        images = self.dit.sample(
            prompt_embeds=conditioned,
            generator=generator,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            **kwargs
        )
        return images
