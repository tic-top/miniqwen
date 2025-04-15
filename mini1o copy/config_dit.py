# --------------------------------------------------------
# Mini1o dit config
# Copyright (c) 2025 Yilin Jia
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
from transformers.configuration_utils import PretrainedConfig
from typing import Optional, Union, Tuple, List
import numpy as np
import copy
###############################################################################
# 1. SanaTransformer2DConfig
###############################################################################
class SanaTransformer2DConfig:
    def __init__(
        self,
        in_channels: int = 32,
        out_channels: Optional[int] = 32,
        num_attention_heads: int = 70,
        attention_head_dim: int = 32,
        num_layers: int = 20,
        num_cross_attention_heads: Optional[int] = 20,
        cross_attention_head_dim: Optional[int] = 112,
        cross_attention_dim: Optional[int] = 2240,
        caption_channels: int = 2304,
        mlp_ratio: float = 2.5,
        dropout: float = 0.0,
        attention_bias: bool = False,
        sample_size: int = 32,
        patch_size: int = 1,
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-6,
        interpolation_scale: Optional[int] = None,
        **kwargs
    ):
        self.in_channels = in_channels
        # 如果 out_channels 为 None，则默认等于 in_channels
        self.out_channels = out_channels or in_channels
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        self.num_layers = num_layers
        self.num_cross_attention_heads = num_cross_attention_heads
        self.cross_attention_head_dim = cross_attention_head_dim
        self.cross_attention_dim = cross_attention_dim
        self.caption_channels = caption_channels
        self.mlp_ratio = mlp_ratio
        self.dropout = dropout
        self.attention_bias = attention_bias
        self.sample_size = sample_size
        self.patch_size = patch_size
        self.norm_elementwise_affine = norm_elementwise_affine
        self.norm_eps = norm_eps
        # 自动计算 interpolation_scale 默认值
        self.interpolation_scale = interpolation_scale if interpolation_scale is not None else max(sample_size // 64, 1)
        self._extra_kwargs = kwargs.copy()    
        
    def to_dict(self):
            # 将当前实例属性拷贝为字典
            output = self.__dict__.copy()
            # 将额外参数（如果有）加入输出中
            extra = output.pop("_extra_kwargs", {})
            output.update(extra)
            return output


###############################################################################
# 2. AutoencoderDCConfig
###############################################################################
class AutoencoderDCConfig:
    def __init__(
        self,
        in_channels: int = 3,
        latent_channels: int = 32,
        attention_head_dim: int = 32,
        encoder_block_types: Union[str, Tuple[str]] = "ResBlock",
        decoder_block_types: Union[str, Tuple[str]] = "ResBlock",
        encoder_block_out_channels: Tuple[int, ...] = (128, 256, 512, 512, 1024, 1024),
        decoder_block_out_channels: Tuple[int, ...] = (128, 256, 512, 512, 1024, 1024),
        encoder_layers_per_block: Tuple[int, ...] = (2, 2, 2, 3, 3, 3),
        decoder_layers_per_block: Tuple[int, ...] = (3, 3, 3, 3, 3, 3),
        encoder_qkv_multiscales: Tuple[Tuple[int, ...], ...] = ((), (), (), (5,), (5,), (5,)),
        decoder_qkv_multiscales: Tuple[Tuple[int, ...], ...] = ((), (), (), (5,), (5,), (5,)),
        upsample_block_type: str = "pixel_shuffle",
        downsample_block_type: str = "pixel_unshuffle",
        decoder_norm_types: Union[str, Tuple[str]] = "rms_norm",
        decoder_act_fns: Union[str, Tuple[str]] = "silu",
        scaling_factor: float = 1.0,
        **kwargs
    ):
        self.in_channels = in_channels
        self.latent_channels = latent_channels
        self.attention_head_dim = attention_head_dim
        self.encoder_block_types = encoder_block_types
        self.decoder_block_types = decoder_block_types
        self.encoder_block_out_channels = encoder_block_out_channels
        self.decoder_block_out_channels = decoder_block_out_channels
        self.encoder_layers_per_block = encoder_layers_per_block
        self.decoder_layers_per_block = decoder_layers_per_block
        self.encoder_qkv_multiscales = encoder_qkv_multiscales
        self.decoder_qkv_multiscales = decoder_qkv_multiscales
        self.upsample_block_type = upsample_block_type
        self.downsample_block_type = downsample_block_type
        self.decoder_norm_types = decoder_norm_types
        self.decoder_act_fns = decoder_act_fns
        self.scaling_factor = scaling_factor
        self._extra_kwargs = kwargs.copy()    
        
    def to_dict(self):
        # 将当前实例属性拷贝为字典
        output = self.__dict__.copy()
        # 将额外参数（如果有）加入输出中
        extra = output.pop("_extra_kwargs", {})
        output.update(extra)
        return output



###############################################################################
# 3. DPMSolverMultistepSchedulerConfig
###############################################################################
class DPMSolverMultistepSchedulerConfig:
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        trained_betas: Optional[Union[np.ndarray, List[float]]] = None,
        solver_order: int = 2,
        prediction_type: str = "epsilon",
        thresholding: bool = False,
        dynamic_thresholding_ratio: float = 0.995,
        sample_max_value: float = 1.0,
        algorithm_type: str = "dpmsolver++",
        solver_type: str = "midpoint",
        lower_order_final: bool = True,
        euler_at_final: bool = False,
        use_karras_sigmas: Optional[bool] = False,
        use_exponential_sigmas: Optional[bool] = False,
        use_beta_sigmas: Optional[bool] = False,
        use_lu_lambdas: Optional[bool] = False,
        use_flow_sigmas: Optional[bool] = False,
        flow_shift: Optional[float] = 1.0,
        final_sigmas_type: Optional[str] = "zero",  # 可选 "zero" 或 "sigma_min"
        lambda_min_clipped: float = -float("inf"),
        variance_type: Optional[str] = None,
        timestep_spacing: str = "linspace",
        steps_offset: int = 0,
        rescale_betas_zero_snr: bool = False,
        **kwargs
    ):
        self.num_train_timesteps = num_train_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_schedule = beta_schedule
        self.trained_betas = trained_betas
        self.solver_order = solver_order
        self.prediction_type = prediction_type
        self.thresholding = thresholding
        self.dynamic_thresholding_ratio = dynamic_thresholding_ratio
        self.sample_max_value = sample_max_value
        self.algorithm_type = algorithm_type
        self.solver_type = solver_type
        self.lower_order_final = lower_order_final
        self.euler_at_final = euler_at_final
        self.use_karras_sigmas = use_karras_sigmas
        self.use_exponential_sigmas = use_exponential_sigmas
        self.use_beta_sigmas = use_beta_sigmas
        self.use_lu_lambdas = use_lu_lambdas
        self.use_flow_sigmas = use_flow_sigmas
        self.flow_shift = flow_shift
        self.final_sigmas_type = final_sigmas_type
        self.lambda_min_clipped = lambda_min_clipped
        self.variance_type = variance_type
        self.timestep_spacing = timestep_spacing
        self.steps_offset = steps_offset
        self.rescale_betas_zero_snr = rescale_betas_zero_snr
        self._extra_kwargs = kwargs.copy()    

    def to_dict(self):
            # 将当前实例属性拷贝为字典
            output = self.__dict__.copy()
            # 将额外参数（如果有）加入输出中
            extra = output.pop("_extra_kwargs", {})
            output.update(extra)
            return output

class DitConfig(PretrainedConfig):
    model_type = "dit"

    def __init__(self,
                 model_config: dict = None,
                 scheduler_config: dict = None,
                 vae_config: dict = None,
                 **kwargs):
        """
        Args:
            model_config (dict, optional): 初始化 SanaTransformer2DModel 的配置参数字典。
            scheduler_config (dict, optional): 初始化 scheduler 的配置参数字典。
            vae_config (dict, optional): 初始化 AutoencoderDC 的配置参数字典。
        """
        super().__init__(**kwargs)
        model_config = model_config or {}
        scheduler_config = scheduler_config or {}
        vae_config = vae_config or {}
        
        self.model_config = SanaTransformer2DConfig(**model_config)
        self.scheduler_config = DPMSolverMultistepSchedulerConfig(**scheduler_config)
        self.vae_config = AutoencoderDCConfig(**vae_config)

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        output["model_config"] = self.model_config.to_dict()
        output["scheduler_config"] = self.scheduler_config.to_dict()
        output["vae_config"] = self.vae_config.to_dict()
        output["model_type"] = self.__class__.model_type
        return output
