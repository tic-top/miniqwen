import math
import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer, AutoProcessor
from transformers import AutoConfig

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values




path = "OpenGVLab/InternVL3-1B"
from mini1o.processor import Mini1oProcessor, Mini1oImageProcessor
from transformers import AutoTokenizer
from diffusers.image_processor import PixArtImageProcessor

image_processor = Mini1oImageProcessor()
gen_image_processor = PixArtImageProcessor()
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
chat_template = "{% set image_count = namespace(value=0) %}{% set video_count = namespace(value=0) %}{% for message in messages %}{% if loop.first and message['role'] != 'system' %}<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n{% endif %}<|im_start|>{{ message['role'] }}\n{% if message['content'] is string %}{{ message['content'] }}<|im_end|>\n{% else %}{% for content in message['content'] %}\n{% if content['type'] == 'image_gen' or 'image_gen' in content %}<|image_gen_start|><|image_gen_pad|><|image_gen_end|>\n{% elif content['type'] == 'video_gen' or 'video_gen' in content %}<|video_gen_start|><|video_gen_pad|><|video_gen_end|>\n{% elif content['type'] == 'image' or 'image' in content or 'image_url' in content %}{% set image_count.value = image_count.value + 1 %}{% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}<|vision_start|><|image_pad|><|vision_end|>\n{% elif content['type'] == 'video' or 'video' in content %}{% set video_count.value = video_count.value + 1 %}{% if add_vision_id %}Video {{ video_count.value }}: {% endif %}<|vision_start|><|video_pad|><|vision_end|>\n{% elif 'text' in content %}{{ content['text'] }}{% endif %}{% endfor %}<|im_end|>\n{% endif %}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}\n"

processor1o = Mini1oProcessor(image_processor=image_processor, 
                              tokenizer=tokenizer, 
                              chat_template=chat_template)
# processor1o.save_pretrained('ckpt')

from PIL import Image
messages = [
    {
        "role": "user",
        "content": [
            {
                "image": Image.open('1.png').convert('RGB'),
            },
            {
                "text": "Please describe the image shortly.\n"
            },
            # {
            #     'image_gen': Image.open('1.png').convert('RGB'),
            # }
        ],
    }
]

# messages = [
#     {
#         "role": "user",
#         "content": [
#             {'text': 'Hello, who are you?'},
#         ]
#     }]

text = processor1o.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
x = processor1o(
    text=[text],
    images=[Image.open('1.png').convert('RGB')],
    return_tensors="pt",
)
print(processor1o.batch_decode(x.input_ids, skip_special_tokens=False)[0])

from mini1o import Mini1oMLLM, Mini1oConfig
import torch

config = Mini1oConfig()
# model = Mini1oMLLM.from_config(config, 
#                                torch_dtype=torch.bfloat16,
#                                use_flash_attn=False,
#                                trust_remote_code=True).to('cuda').eval()
model = Mini1oMLLM.from_pretrained('ckpt', 
                               mllm_pretrained_path='OpenGVLab/InternVL2_5-1B-MPO',
                               torch_dtype=torch.bfloat16,
                               use_flash_attn=False,
                               trust_remote_code=True).to('cuda').eval()

x = {k: v.to('cuda:0') for k, v in x.items() if isinstance(v, torch.Tensor)}
x['pixel_values'] = x['pixel_values'].to(torch.bfloat16)
# x['pixel_values'] = load_image('1.png', max_num=12).to(torch.bfloat16).cuda()
# print(x)
with torch.no_grad():
    output = model.generate(**x, max_new_tokens=200, do_sample=True)
    print(processor1o.batch_decode(output, skip_special_tokens=False)[0])