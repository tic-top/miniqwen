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
    from config import Mini1oConfig
except:
    from .dit import Mini1oDiT
    from .mllm import Mini1oMLLM
    from .config import Mini1oConfig


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
        mllm_out = self.mllm(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )

        hidden_state = mllm_out.hidden_states[-1]
        condition_mask = mllm_out.condition_mask
        condition_tokens= hidden_state.reshape(-1, self.hidden_dim)[condition_mask].reshape(-1,self.num_img_gen_tokens, self.hidden_dim)
        # 3. 通过 connector 将 mllm 得到的条件信息映射到扩散模型的条件嵌入空间
        conditioned = self.connector(condition_tokens)
        image_loss = self.dit(
            clean_image=clean_image, # [n * 3 * H * W]
            prompt_embeds=conditioned,
            **kwargs
        )
        return mllm_out.loss, image_loss

    @torch.no_grad()
    def generate(self, pixel_values: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor, generator: torch.Generator = None, num_inference_steps: int = 50, guidance_scale: float = 1.0, **kwargs) -> torch.Tensor:
        return self.mllm.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            generator=generator,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            **kwargs
        )
    
    # from pretrained
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """
        从预训练模型加载参数，并初始化 Mini1o 模型。

        参数：
            pretrained_model_name_or_path (str): 预训练模型的路径或名称。
            **kwargs: 其他传入参数。
        """
        # 加载预训练模型的配置
        config = Mini1oConfig.from_pretrained(pretrained_model_name_or_path)
        # 初始化模型
        model = cls(config, **kwargs)
        # 加载预训练权重
        model.load_state_dict(torch.load(pretrained_model_name_or_path), strict=False)
        return model