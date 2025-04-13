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
try:
    from diffloss import Diffloss
except:
    from .diffloss import Diffloss

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
        ## Diffusion loss
        self.diff_loss = Diffloss(diffusion_config)
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
        gen_pixel_values: Optional[torch.FloatTensor] = None, # num_images x 3 x 64 x 64
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
        text_loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_logits = shift_logits.view(-1, self.language_model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            text_loss = F.cross_entropy(shift_logits, shift_labels, reduction="mean")
        image_loss = None
        if gen_pixel_values is not None:
            # get the image features from the hidden states
            condition_features = hidden_states[-1][image_generation_mask]
            condition_features = condition_features.reshape(B, num_image_generation_tokens, C)
            condition_features = self.connector(condition_features)

            # pass the diffusion model to get the output feature
            image_loss = self.diffusion_model(
                clean_image=gen_pixel_values,
                prompt_embeds=condition_features,
                return_dict=True,
            )

        if text_loss is not None and image_loss is not None:
            print('error, now just train the image')
            loss = text_loss + image_loss
        elif text_loss is not None:
            print('error, now just train the image')
            loss = text_loss
        elif image_loss is not None:
            loss = image_loss

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=hidden_states,
            attentions=outputs.attentions,
        )
    
    @torch.no_grad()
    def generate(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        visual_features: Optional[torch.FloatTensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        output_hidden_states: Optional[bool] = None,
        **generate_kwargs,
    ) -> torch.Tensor:
        """
        生成过程：
          - 如果输入中包含特殊 token (<image_gen_start>，id == 15000)，
            则先正常生成一段并截取前缀，然后利用 self.image_gen_queries
            迭代生成图像条件部分，每个 query 依次喂入生成一小段，最后生成一个 <image_gen_end> token，
            将这些生成结果拼接到前缀后，再作为 prompt 继续生成后续文本。
          - 否则直接走正常生成流程。
        """
        # 通过 pixel_values 提取视觉特征（如果有的话），插入到文本 embedding 对应的位置
        if pixel_values is not None:
            if visual_features is not None:
                vit_embeds = visual_features
            else:
                # 这里假设你有 extract_feature 方法
                vit_embeds = self.extract_feature(pixel_values)
            # 获取 LLM 的输入 embeddings
            inputs_embeds = self.mllm.get_input_embeddings()(input_ids)
            B, N, C = inputs_embeds.shape
            inputs_embeds = inputs_embeds.view(B * N, C)
            input_ids_flat = input_ids.view(B * N)
            # 找到 context token 位置，并替换为视觉特征
            selected = (input_ids_flat == self.img_context_token_id)
            if selected.sum() == 0:
                raise ValueError("没有检测到 img_context_token_id")
            inputs_embeds[selected] = vit_embeds.view(-1, C).to(inputs_embeds.device)
            inputs_embeds = inputs_embeds.view(B, N, C)
        else:
            inputs_embeds = self.mllm.get_input_embeddings()(input_ids)
        
        # 检查是否包含特殊的 <image_gen_start>
        if input_ids is not None and (input_ids == self.image_gen_start_token_id).any():
            pass
        else:
            # 如果没有特殊 token，则直接走常规生成流程
            outputs = self.mllm.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                generation_config=generation_config,
                output_hidden_states=output_hidden_states,
                use_cache=True,
                **generate_kwargs,
            )
            return outputs