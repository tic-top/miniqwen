import os
import json
from typing import Union, List, Dict, Any, Optional
import torch
import numpy as np
from PIL import Image
from transformers import ProcessorMixin, ImageProcessingMixin
from transformers.image_processing_utils import (
    BaseImageProcessor,
    BatchFeature,
    get_size_dict,
)
from transformers.image_transforms import (
    convert_to_rgb,
    resize,
    to_channel_dimension_format,
)
from transformers.image_utils import ImageInput, VideoInput
from transformers.processing_utils import ImagesKwargs, ProcessingKwargs, ProcessorMixin, Unpack, VideosKwargs
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput
from transformers.utils import (
    TensorType,
    add_start_docstrings,
    is_torch_available,
    is_torchvision_available,
    is_torchvision_v2_available,
    is_vision_available,
    logging,
)

if is_torchvision_available():
    if is_torchvision_v2_available():
        from torchvision.transforms.v2 import functional as F
    else:
        from torchvision.transforms import functional as F

from diffusers.image_processor import PixArtImageProcessor
from transformers import AutoProcessor 
AutoProcessor.register(PixArtImageProcessor, "PixArtImageProcessor")

##########################################
# Mini1o 图像预处理器
# 该处理器参考了前面代码，实现图像的加载、动态预处理和分块。
    
class Mini1oImageProcessor(BaseImageProcessor):
    r"""
    Mini1oImageProcessor 用于将输入图像进行处理、调整大小、动态分块、归一化，
    并返回模型所需的像素张量及（可选）相关图像信息。

    参数:
        input_size (int, optional): 每个分块的边长，默认 448.
        max_num (int, optional): 最大分块数量，默认 12.
        use_thumbnail (bool, optional): 若图像分块不为 1，则是否额外附加缩略图，默认 True.
        imagenet_mean (tuple, optional): 归一化均值，默认 (0.485, 0.456, 0.406).
        imagenet_std (tuple, optional): 归一化标准差，默认 (0.229, 0.224, 0.225).
        do_center_crop (bool, optional): 是否执行中心裁剪，默认 False.
        crop_size (int or tuple, optional): 中心裁剪的尺寸，若为 int 则宽高一致，默认 None.
    """
    def __init__(
        self,
        size=(448, 448),
        max_num= 12,
        use_thumbnail = True,
        image_mean = (0.485, 0.456, 0.406),
        image_std = (0.229, 0.224, 0.225),
        do_center_crop = False,
        do_convert_rgb = True,
        crop_size = None,
        resample=Image.Resampling.BILINEAR,
    ):
        self.size = size
        self.crop_size = crop_size
        self.input_size = size[0]
        self.max_num = max_num
        self.use_thumbnail = use_thumbnail
        self.image_mean = image_mean
        self.image_std = image_std

        # 标记处理步骤的是否执行
        self.do_resize = True
        self.resample = resample
        self.do_center_crop = do_center_crop
        self.do_normalize = True
        self.do_convert_rgb = do_convert_rgb

    def preprocess(self, images):
        if self.do_resize:
            images = [resize(image=image, size=self.size, resample=self.resample) for image in images]
        if self.do_center_crop and self.crop_size is not None:
            images = [self.center_crop(image, self.crop_size) for image in images]
        if self.do_normalize:
            images = [self.normalize(image=image, mean=self.image_mean, std=self.image_std) for image in images]
        return images

    @staticmethod
    def find_closest_aspect_ratio(aspect_ratio: float, target_ratios: List[tuple],
                                  width: int, height: int, image_size: int) -> tuple:
        """
        在目标宽高比集合中，找出与图像宽高比最接近的比例。
        """
        best_ratio_diff = float('inf')
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect = ratio[0] / ratio[1]
            diff = abs(aspect_ratio - target_aspect)
            if diff < best_ratio_diff:
                best_ratio_diff = diff
                best_ratio = ratio
            elif diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio

    def dynamic_preprocess(self, image: Image.Image) -> List[Image.Image]:
        """
        动态预处理图像：
          1. 根据图像宽高比选择最佳目标比例；
          2. 按目标比例调整图像尺寸并分块；
          3. 当 use_thumbnail 为 True 且分块数不为 1 时，添加原图缩略图。
        """
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        target_ratios = set(
            (i, j)
            for n in range(1, self.max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if i * j <= self.max_num and i * j >= 1
        )
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        best_ratio = self.find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, self.input_size)
        target_width = int(self.input_size * best_ratio[0])
        target_height = int(self.input_size * best_ratio[1])
        num_blocks = best_ratio[0] * best_ratio[1]

        resized_img = image.resize((target_width, target_height))
        processed_images = []
        num_blocks_horizontal = target_width // self.input_size

        for i in range(num_blocks):
            left = (i % num_blocks_horizontal) * self.input_size
            upper = (i // num_blocks_horizontal) * self.input_size
            split_img = resized_img.crop((left, upper, left + self.input_size, upper + self.input_size))
            processed_images.append(split_img)

        if self.use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((self.input_size, self.input_size))
            processed_images.append(thumbnail_img)
        return processed_images

    def _load_and_preprocess(self, image: Union[str, Image.Image]) -> torch.Tensor:
        """
        加载图像（支持文件路径或 PIL.Image），调用动态预处理，
        然后将每个分块转换为 tensor 并堆叠。
        """
        if isinstance(image, str):
            image = Image.open(image) # now Image.Image
        if self.do_convert_rgb:
            image = convert_to_rgb(image)
        patches = self.dynamic_preprocess(image) #image: Image.Image
        # from Image.Image to numpy array
        patches = [np.array(patch)/255.0 for patch in patches]
        
        pixel_values = self.preprocess(patches) # List[np.array]
        pixel_values = [torch.tensor(np.array(patch)).permute(2, 0, 1) for patch in pixel_values]
        return torch.stack(pixel_values)

    def __call__(self, images: Union[Image.Image, str, List[Union[Image.Image, str]]], **kwargs) -> Dict[str, Any]:
        """
        支持单张或多张图像输入，返回一个包含 "pixel_values" 键的字典。
        """
        if not isinstance(images, list):
            images = [images]
        all_pixel_values = []
        num_patches_list = []
        for img in images:
            pixel_tensor = self._load_and_preprocess(img)
            all_pixel_values.append(pixel_tensor)
            num_patches_list.append(pixel_tensor.shape[0])
        # 这里将所有图像的 tensor 在第0维拼接；后续模型可根据 num_patches_list 分别处理
        pixel_values = torch.cat(all_pixel_values, dim=0)
        return {"pixel_values": pixel_values, "num_patches_list": num_patches_list}


##########################################
# 整体 Mini1oProcessor
# 该处理器负责将图像和文本处理结合在一起，生成模型输入，同时提供对生成输出的后处理

class Mini1oProcessor(ProcessorMixin):
    r"""
    Mini1oProcessor 将 Mini1oImageProcessor 和文本 tokenizer 封装到一起，用于构造多模态模型的输入，
    并提供对输出的后处理方法。该 processor 仅负责生成输入与处理输出，不直接调用生成方法。

    参数:
        image_processor ([`Mini1oImageProcessor`]): 用于图像预处理。
        gen_image_processor: vae_scale_factor= 0.41407
        tokenizer: 文本 tokenizer 对象。
        chat_template (str, optional): 可选的对话模板，用于组织多轮对话文本（内部使用 Jinja 或自定义模板）。
        system_message (str, optional): 系统提示信息，将在构造对话输入时自动添加。
    """
    attributes = ["image_processor", "tokenizer"]
    valid_kwargs = ["chat_template"]

    image_processor_class = "AutoImageProcessor"
    tokenizer_class = ("Qwen2Tokenizer", "Qwen2TokenizerFast")

    def __init__(self, image_processor: Mini1oImageProcessor, tokenizer, chat_template: Union[str, None] = None, system_message: str = "", **kwargs):
        self.image_pad_token = "<|image_pad|>" if not hasattr(tokenizer, "image_token") else tokenizer.image_token
        self.video_pad_token = "<|video_pad|>" if not hasattr(tokenizer, "video_token") else tokenizer.video_token
        self.image_gen_token = "<|image_gen_pad|>" if not hasattr(tokenizer, "image_gen_token") else tokenizer.image_gen_token
        self.video_gen_token = "<|video_gen_pad|>" if not hasattr(tokenizer, "video_gen_token") else tokenizer.video_gen_token
        self.num_image_gen_token = 64
        super().__init__(image_processor, tokenizer, chat_template=chat_template)
        
    def __call__(
        self,
        images: Union[Image.Image, str, List[Union[Image.Image, str]]] = None,
        gen_images: Union[Image.Image, str, List[Union[Image.Image, str]]] = None,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        **kwargs,
    ) -> BatchFeature:

        """
        构造模型输入字典，包括文本的 token 化结果和图像预处理后的像素张量。
        
        如果同时提供图像和文本，本方法会检查文本中是否包含图像占位符 `<image>`，
        若未包含且 images 不为空，则默认在文本前添加 `<image>\n` 前缀。

        返回字典通常包含：
            - input_ids, attention_mask 等文本输入相关字段；
            - pixel_values：图像 tensor；
            - num_patches_list：各图像对应的 patch 数量列表。
        """
        if images is not None:
            image_outputs = self.image_processor(images=images)
            image_num_patches_list = image_outputs["num_patches_list"]
        else:
            image_outputs = {}
            image_num_patches_list = None
        if gen_images is not None:
            # image_gen_outputs = self.gen_image_processor(images=gen_images)
            pass
        else:
            image_gen_outputs = {}

        if not isinstance(text, list):
            text = [text]
        
        if image_num_patches_list is not None:
            index = 0
            for i in range(len(text)):
                while self.image_pad_token in text[i]:
                    text[i] = text[i].replace(self.image_pad_token, "<|placeholder|>" * image_num_patches_list[index] * 256, 1)
                    index += 1
                text[i] = text[i].replace("<|placeholder|>", self.image_pad_token)
            assert index == len(image_num_patches_list), "num of image pad and num of image are different"
        
        if True:
            index = 0
            for i in range(len(text)):
                while self.image_gen_token in text[i]:
                    text[i] = text[i].replace(self.image_gen_token, "<|placeholder|>" * self.num_image_gen_token, 1)
                    index += 1
                text[i] = text[i].replace("<|placeholder|>", self.image_gen_token)

        text_inputs = self.tokenizer(text, **kwargs)

        return BatchFeature(data={**text_inputs, **image_outputs, **image_gen_outputs})
    
    def post_process(self, generated_outputs, skip_special_tokens = True, clean_up_tokenization_spaces = False, **kwargs) -> List[str]:
        """
        对生成模型输出的 token id 序列进行后处理，返回文本字符串。
        通常调用 tokenizer 的 batch_decode 方法。
        """
        return self.tokenizer.batch_decode(
            generated_outputs,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            **kwargs,
        )

    def batch_decode(self, *args, **kwargs) -> List[str]:
        """
        调用 tokenizer 的 batch_decode 方法。 
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs) -> str:
        """
        调用 tokenizer 的 decode 方法。 
        """
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self) -> List[str]:
        """
        返回模型输入字典中所有可能包含的键名，通常为 tokenizer 和 image_processor 的输入名称合集。
        """
        tokenizer_input_names = self.tokenizer.model_input_names if hasattr(self.tokenizer, "model_input_names") else []
        image_processor_keys = ["pixel_values", "num_patches_list"]
        diffusion_image_processor_keys = []
        return list(dict.fromkeys(tokenizer_input_names + image_processor_keys, diffusion_image_processor_keys))

