import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from torchvision import transforms
from PIL import Image
import numpy as np
from typing import Optional, Dict, Any, List
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from transformers.utils import ModelOutput

@dataclass
class CausalULMOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    condition_states: Optional[Tuple[torch.FloatTensor, ...]] = None

class Connector(nn.Module):
    def __init__(self, hidden_dim, diffusion_dim, num_layers=6, nhead=8):
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
    
class Mini1o(PreTrainedModel):
    def __init__(self, mllm_config, diffusion_config):
        ## -- load the mllm model -- ##
        self.mllm = AutoModelForCausalLM(mllm_config)
        ## -- set the meta querys for image generation-- ##
        self.hidden_dim = self.mllm.config.hidden_size
        self.num_image_gen_tokens = self.mllm_config.num_image_gen_tokens
        self.image_gen_queries = nn.Parameter(torch.randn(self.num_image_gen_tokens, self.hidden_dim))
        ## -- set the mllm and dit connector-- ##
        self.connector = Connector(hidden_dim=mllm_config['hidden_dim'], diffusion_dim=diffusion_config['diffusion_dim'])
        self.img_context_token_id = mllm_config.get("img_context_token_id", 10000)  # 如用于 ViT 特征插入
        self.image_gen_start_token_id = mllm_config.get("image_gen_start_token_id", 15000)  # 表示开始图像生成
        self.image_gen_end_token_id = mllm_config.get("image_gen_end_token_id", 15001)      # 表示结束图像生成

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        image_flags: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        if inputs_embeds is None:
            ## -- get the text embeddings -- ##
            inputs_embeds = self.mllm.language_model.get_input_embeddings()(input_ids).clone()
            B, N, C = inputs_embeds.shape
            inputs_embeds = inputs_embeds.reshape(B * N, C)

            ## -- get the image embeddings -- ##
            image_flags = image_flags.squeeze(-1)
            vit_embeds = self.mllm.extract_feature(pixel_values)
            vit_embeds = vit_embeds[image_flags == 1]
            vit_batch_size = pixel_values.shape[0]
            if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
                print(f'dynamic ViT batch size: {vit_batch_size}, images per sample: {vit_batch_size / B}, dynamic token length: {N}')

            ## -- insert the image embeddings into the input embeddings -- ##
            input_ids = input_ids.reshape(B * N)
            image_mask = (input_ids == self.img_context_token_id)
            try:
                inputs_embeds[image_mask] = inputs_embeds[image_mask] * 0.0 + vit_embeds.reshape(-1, C)
            except Exception as e:
                vit_embeds = vit_embeds.reshape(-1, C)
                print(f'warning: {e}, inputs_embeds[image_mask].shape={inputs_embeds[image_mask].shape}, '
                    f'vit_embeds.shape={vit_embeds.shape}')
                n_token = image_mask.sum()
                inputs_embeds[image_mask] = inputs_embeds[image_mask] * 0.0 + vit_embeds[:n_token]
            
            ## -- insert the meta queries into the input embeddings -- ##
            image_generation_mask = (input_ids == self.image_generation_token_id)
            num_image_generation_tokens = image_generation_mask.sum().item()
            # n = gen_pixel_values.shape[0]
            n = num_image_generation_tokens // self.num_image_gen_tokens
            num_image_generation_features = self.num_image_gen_tokens * n# 64 * num of images
            if num_image_generation_tokens != num_image_generation_features:
                raise ValueError(f"num_image_generation_tokens: {num_image_generation_tokens}, "
                                 f"num_image_generation_features: {num_image_generation_features}")
            
            repeated_queries = self.image_gen_queries.unsqueeze(0).repeat(n, 1, 1).reshape(-1, C)
            inputs_embeds[image_generation_mask] = repeated_queries
            inputs_embeds = inputs_embeds.reshape(B, N, C)

        outputs = self.mllm.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        logits = outputs.logits
        hidden_states = outputs.hidden_states

        # get the generated image features fromt the hidden states
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_logits = shift_logits.view(-1, self.language_model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = F.cross_entropy(shift_logits, shift_labels, reduction="mean")
        if (input_ids == self.image_generation_token_id).sum().item():
            # get the image features from the hidden states
            condition_features = hidden_states[-1][image_generation_mask]
            condition_features = condition_features.reshape(B, num_image_generation_tokens, C)
            condition_features = self.connector(condition_features)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalULMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=hidden_states,
            attentions=outputs.attentions,
            condition_states=condition_features,
        )
    

    @torch.no_grad()
    def generate(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,
            input_ids: Optional[torch.FloatTensor] = None,
            input_embeds: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.LongTensor] = None,
            visual_features: Optional[torch.FloatTensor] = None,
            generation_config: Optional[GenerationConfig] = None,
            output_hidden_states: Optional[bool] = None,
            **generate_kwargs,
    ) -> torch.LongTensor:

        assert self.img_context_token_id is not None
        if pixel_values is not None:
            if visual_features is not None:
                vit_embeds = visual_features
            else:
                vit_embeds = self.extract_feature(pixel_values)
            input_embeds = self.language_model.get_input_embeddings()(input_ids)
            B, N, C = input_embeds.shape
            input_embeds = input_embeds.reshape(B * N, C)

            input_ids = input_ids.reshape(B * N)
            selected = (input_ids == self.img_context_token_id)
            assert selected.sum() != 0
            input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)

            input_embeds = input_embeds.reshape(B, N, C)
        else:
            input_embeds = self.language_model.get_input_embeddings()(input_ids)

        outputs = self.language_model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            generation_config=generation_config,
            output_hidden_states=output_hidden_states,
            use_cache=True,
            **generate_kwargs,
        )

        return outputs