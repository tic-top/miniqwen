# # --------------------------------------------------------
# # InternVL
# # Copyright (c) 2024 OpenGVLab
# # Licensed under The MIT License [see LICENSE for details]
# # --------------------------------------------------------

# import copy

# from transformers import AutoConfig, LlamaConfig, Qwen2Config
# from transformers.configuration_utils import PretrainedConfig
# from transformers.utils import logging

# from .config_vit import InternVisionConfig
# from .config_dit import DitConfig

# logger = logging.get_logger(__name__)


# class Mini1oConfig(PretrainedConfig):
#     model_type = 'mini1o'
#     is_composition = True

#     def __init__(
#             self,
#             dit_config=None,
#             vision_config=None,
#             llm_config=None,
#             use_backbone_lora=0,
#             use_llm_lora=0,
#             select_layer=-1,
#             force_image_size=None,
#             downsample_ratio=0.5,
#             template=None,
#             dynamic_image_size=False,
#             use_thumbnail=False,
#             ps_version='v1',
#             min_dynamic_patch=1,
#             max_dynamic_patch=6,
#             num_img_gen_tokens=64,
#             img_gen_start_token_id=114514, ## start of image generation
#             img_gen_context_token_id=114514, ## pad for image generation
#             img_gen_end_token_id=114514, ## end of image generation
#             img_context_token_id=151655, ## 这个是image pad
#             **kwargs):
#         super().__init__(**kwargs)

#         if dit_config is None:
#             dit_config = {}
#             logger.info('dit_config is None. Initializing the DitConfig with default values.')

#         if vision_config is None:
#             vision_config = {'architectures': ['InternVisionModel']}
#             logger.info('vision_config is None. Initializing the InternVisionConfig with default values.')

#         if llm_config is None:
#             llm_config = {'architectures': ['Qwen2ForCausalLM']}
#             logger.info('llm_config is None. Initializing the LlamaConfig config with default values (`LlamaConfig`).')

#         self.vision_config = InternVisionConfig(**vision_config)
#         self.dit_config = DitConfig(**dit_config)

#         if llm_config.get('architectures')[0] == 'LlamaForCausalLM':
#             self.llm_config = LlamaConfig(**llm_config)
#         elif llm_config.get('architectures')[0] == 'Qwen2ForCausalLM':
#             self.llm_config = Qwen2Config(**llm_config)
#         else:
#             raise ValueError('Unsupported architecture: {}'.format(llm_config.get('architectures')[0]))
#         self.use_backbone_lora = use_backbone_lora
#         self.use_llm_lora = use_llm_lora
#         self.select_layer = select_layer
#         self.force_image_size = force_image_size
#         self.downsample_ratio = downsample_ratio
#         self.template = template
#         self.dynamic_image_size = dynamic_image_size
#         self.use_thumbnail = use_thumbnail
#         self.ps_version = ps_version  # pixel shuffle version
#         self.min_dynamic_patch = min_dynamic_patch
#         self.max_dynamic_patch = max_dynamic_patch
#         # By default, we use tie_word_embeddings=False for models of all sizes.
#         self.tie_word_embeddings = self.llm_config.tie_word_embeddings

#         # Image generation related
#         self.num_img_gen_tokens=num_img_gen_tokens
#         self.img_gen_start_token_id=img_gen_start_token_id
#         self.img_gen_context_token_id=img_gen_context_token_id
#         self.img_gen_end_token_id=img_gen_end_token_id
#         self.img_context_token_id=img_context_token_id

#         logger.info(f'vision_select_layer: {self.select_layer}')
#         logger.info(f'ps_version: {self.ps_version}')
#         logger.info(f'min_dynamic_patch: {self.min_dynamic_patch}')
#         logger.info(f'max_dynamic_patch: {self.max_dynamic_patch}')

#     def to_dict(self):
#         """
#         Serializes this instance to a Python dictionary. Override the default [`~PretrainedConfig.to_dict`].
#         Returns:
#             `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
#         """
#         output = copy.deepcopy(self.__dict__)
#         output['dit_config'] = self.dit_config.to_dict()
#         output['vision_config'] = self.vision_config.to_dict()
#         output['llm_config'] = self.llm_config.to_dict()
#         output['model_type'] = self.__class__.model_type
#         output['use_backbone_lora'] = self.use_backbone_lora
#         output['use_llm_lora'] = self.use_llm_lora
#         output['select_layer'] = self.select_layer
#         output['force_image_size'] = self.force_image_size
#         output['downsample_ratio'] = self.downsample_ratio
#         output['template'] = self.template
#         output['dynamic_image_size'] = self.dynamic_image_size
#         output['use_thumbnail'] = self.use_thumbnail
#         output['ps_version'] = self.ps_version
#         output['min_dynamic_patch'] = self.min_dynamic_patch
#         output['max_dynamic_patch'] = self.max_dynamic_patch

#         return output
