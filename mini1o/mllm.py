import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, List, Tuple
from transformers import AutoModelForCausalLM, GenerationConfig, PreTrainedModel
from transformers.utils import ModelOutput
from .config import Mini1oConfig

# 自定义 ModelOutput，增加 condition_mask 用于标记特殊处理位置
@dataclass
class CausalULMOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    condition_mask: Optional[torch.Tensor] = None

class Mini1oMLLM(PreTrainedModel):
    config_class = Mini1oConfig  # 指定自定义配置
    
    def __init__(self, config: Mini1oConfig, **kwargs):
        super().__init__(config)
        self.config = config

        # 从预训练模型加载基础 mllm（例如一个 causal MLLM 模型）
        self.mllm = AutoModelForCausalLM.from_config(config.mllm_config, trust_remote_code=True, **kwargs)

        # 记录特殊 token id 参数
        self.num_img_gen_tokens = config.num_img_gen_tokens
        self.img_context_token_id = config.img_context_token_id
        self.img_gen_start_token_id = config.img_gen_start_token_id
        self.img_gen_context_token_id = config.img_gen_context_token_id
        self.img_gen_end_token_id = config.img_gen_end_token_id

        # 依据基础模型 hidden_size 来初始化 meta query 参数
        self.hidden_dim = self.mllm.config.hidden_size
        self.img_gen_queries = nn.Parameter(torch.randn(self.num_img_gen_tokens, self.hidden_dim))

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
            # 获取文本的 embeddings
            inputs_embeds = self.mllm.language_model.get_input_embeddings()(input_ids).clone()
            B, N, C = inputs_embeds.shape
            inputs_embeds = inputs_embeds.view(B * N, C)

            # 获取 image embeddings（依据 image_flags 决定哪些样本需要替换）
            image_flags = image_flags.squeeze(-1)
            vit_embeds = self.mllm.extract_feature(pixel_values)
            vit_embeds = vit_embeds[image_flags == 1]
            if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
                vit_batch_size = pixel_values.shape[0]
                print(f'dynamic ViT batch size: {vit_batch_size}, images per sample: {vit_batch_size / B}, dynamic token length: {N}')

            # 替换输入 embedding 中标记为图像上下文的 token（img_context_token_id）
            input_ids_flat = input_ids.view(B * N)
            image_mask = (input_ids_flat == self.img_context_token_id)
            try:
                inputs_embeds[image_mask] = vit_embeds.view(-1, C)
            except Exception as e:
                n_token = image_mask.sum()
                inputs_embeds[image_mask] = vit_embeds.view(-1, C)[:n_token]
            
            # 替换为 meta query（当 token id 与 img_gen_context_token_id 匹配时）
            img_generation_mask = (input_ids_flat == self.img_gen_context_token_id)
            num_img_generation_tokens = img_generation_mask.sum().item()
            n = num_img_generation_tokens // self.num_img_gen_tokens
            num_img_generation_features = self.num_img_gen_tokens * n
            if num_img_generation_tokens != num_img_generation_features:
                raise ValueError(f"num_img_generation_tokens: {num_img_generation_tokens}, "
                                 f"expected multiple of {self.num_img_gen_tokens}")
            repeated_queries = self.img_gen_queries.unsqueeze(0).repeat(n, 1, 1).view(-1, C)
            inputs_embeds[img_generation_mask] = repeated_queries
            inputs_embeds = inputs_embeds.view(B, N, C)
        
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
        
        loss = None
        if labels is not None:
            # Shift logits 和 labels 以计算交叉熵 loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_logits = shift_logits.view(-1, self.mllm.language_model.config.vocab_size)
            shift_labels = shift_labels.view(-1).to(shift_logits.device)
            loss = F.cross_entropy(shift_logits, shift_labels, reduction="mean")
        
        if not return_dict:
            out = (logits,) + outputs[1:]
            return (loss,) + out if loss is not None else out
        
        return CausalULMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=hidden_states,
            attentions=outputs.attentions,
            condition_mask=img_generation_mask if 'img_generation_mask' in locals() else None,
        )
    
    @torch.no_grad()
    def generate(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        visual_features: Optional[torch.FloatTensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        output_hidden_states: Optional[bool] = None,
        **generate_kwargs,
    ) -> torch.LongTensor:
        # 负数标记的 token 用于 meta query 替换
        img_gen_context_token_mask = input_ids < 0
        input_ids = input_ids.abs()
        
        if pixel_values is not None:
            # 根据 pixel_values 获取 image embeddings
            vit_embeds = visual_features if visual_features is not None else self.mllm.extract_feature(pixel_values)
            input_embeds = self.mllm.language_model.get_input_embeddings()(input_ids)
            B, N, C = input_embeds.shape
            input_embeds = input_embeds.view(B * N, C)

            input_ids_flat = input_ids.view(B * N)
            selected = (input_ids_flat == self.img_context_token_id)
            if selected.sum() == 0:
                raise ValueError("未找到 image context token")
            input_embeds[selected] = vit_embeds.view(-1, C).to(input_embeds.device)
            input_embeds = input_embeds.view(B, N, C)
        else:
            input_embeds = self.mllm.language_model.get_input_embeddings()(input_ids)
        
        if img_gen_context_token_mask.any():
            # 将负数 token 映射到 meta query：例如 -1 对应 index 0，-2 对应 index 1，依此类推
            meta_idx = (input_ids - 1)[img_gen_context_token_mask]
            if meta_idx.max() >= self.img_gen_queries.shape[0]:
                raise ValueError("meta token 超出预定义 metaquery 长度")
            input_embeds[img_gen_context_token_mask] = self.img_gen_queries[meta_idx].to(input_embeds.device)
        
        outputs = self.mllm.language_model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            generation_config=generation_config,
            output_hidden_states=output_hidden_states,
            use_cache=True,
            **generate_kwargs,
        )
        return outputs

    @classmethod
    def from_config(cls, config: Mini1oConfig, **kwargs):
        return cls(config, **kwargs)