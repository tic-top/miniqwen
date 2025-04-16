# --------------------------------------------------------
# Mini2o config
# Copyright (c) 2025 Yilin Jia
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import copy
from transformers import AutoConfig
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)

class ConnectorConfig:
    def __init__(
        self,
        hidden_dim: int = 896,
        diffusion_dim: int = 2304,
        num_layers: int = 6,
        nhead: int = 8,
        **kwargs
    ):
        self.hidden_dim = hidden_dim
        self.diffusion_dim = diffusion_dim
        self.num_layers = num_layers
        self.nhead = nhead
        self._extra_kwargs = kwargs.copy()

    def to_dict(self):
        # 将当前实例属性拷贝为字典
        output = self.__dict__.copy()
        # 将额外参数（如果有）加入输出中
        extra = output.pop("_extra_kwargs", {})
        output.update(extra)
        return output

class Mini2oConfig(PretrainedConfig):
    model_type = 'mini2o'
    is_composition = True

    def __init__(
            self,
            mllm_config=None,
            connector_config=None,
            num_img_gen_tokens=256,
            img_gen_start_token_id=114514, ## start of image generation
            img_gen_context_token_id=114514, ## pad for image generation
            img_gen_end_token_id=114514, ## end of image generation
            img_context_token_id=151655, ## 这个是image pad
            **kwargs):
        super().__init__(**kwargs)


        if mllm_config is None:
            mllm_config = {'_name_or_path': "OpenGVLab/InternVL3-1B"}
            logger.info('mllm_config is None. Initializing the MLLMConfig with default values.')

        self.mllm_config = AutoConfig.from_pretrained(mllm_config['_name_or_path'], trust_remote_code=True)

        if connector_config is None:
            # self.dit_config.model_config.caption_channels
            connector_config = {'hiffen_dim': self.mllm_config.llm_config.hidden_size, 
                                'diffusion_dim': 2304, 
                                'num_layers': 6, 
                                'nhead': 8}
            logger.info('connector_config is None. Initializing the ConnectorConfig with default values.')
    
        self.connector_config = ConnectorConfig(**connector_config)

        # Image generation related
        self.num_img_gen_tokens=num_img_gen_tokens
        self.img_gen_start_token_id=img_gen_start_token_id
        self.img_gen_context_token_id=img_gen_context_token_id
        self.img_gen_end_token_id=img_gen_end_token_id
        self.img_context_token_id=img_context_token_id
        self.diffusion_loss_weight=2/10


    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default [`~PretrainedConfig.to_dict`].
        Returns:
            `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__dict__)
        output['mllm_config'] = self.mllm_config.to_dict()
        output['connector_config'] = self.connector_config.to_dict()
        return output
