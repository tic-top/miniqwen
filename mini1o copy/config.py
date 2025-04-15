# --------------------------------------------------------
# Mini1o config
# Copyright (c) 2025 Yilin Jia
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import copy
from transformers import AutoConfig
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging
from .config_dit import DitConfig

logger = logging.get_logger(__name__)


class Mini1oConfig(PretrainedConfig):
    model_type = 'mini1o'
    is_composition = True

    def __init__(
            self,
            mllm_config=None,
            dit_config=None,
            num_img_gen_tokens=64,
            img_gen_start_token_id=114514, ## start of image generation
            img_gen_context_token_id=114514, ## pad for image generation
            img_gen_end_token_id=114514, ## end of image generation
            img_context_token_id=151655, ## 这个是image pad
            **kwargs):
        super().__init__(**kwargs)

        if dit_config is None:
            dit_config = {}
            logger.info('dit_config is None. Initializing the DitConfig with default values.')

        if mllm_config is None:
            mllm_config = {'_name_or_path': "OpenGVLab/InternVL2_5-1B-MPO"}
            logger.info('mllm_config is None. Initializing the MLLMConfig with default values.')

        self.mllm_config = AutoConfig.from_pretrained(mllm_config['_name_or_path'], trust_remote_code=True)
        self.dit_config = DitConfig(**dit_config)

        # Image generation related
        self.num_img_gen_tokens=num_img_gen_tokens
        self.img_gen_start_token_id=img_gen_start_token_id
        self.img_gen_context_token_id=img_gen_context_token_id
        self.img_gen_end_token_id=img_gen_end_token_id
        self.img_context_token_id=img_context_token_id

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default [`~PretrainedConfig.to_dict`].
        Returns:
            `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__dict__)
        output['dit_config'] = self.dit_config.to_dict()
        output['mllm_config'] = self.mllm_config.to_dict()
        return output
