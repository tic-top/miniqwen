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
                "image": Image.open('sana.png').convert('RGB'),
            },
            {
                "text": "Please describe the image shortly.\n"
            },
            # {
            #     'image_gen': Image.open('sana.png').convert('RGB'),
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
    images=[Image.open('sana.png').convert('RGB')],
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
                               mllm_pretrained_path='OpenGVLab/InternVL3-1B',
                               torch_dtype=torch.bfloat16,
                               use_flash_attn=False,
                               trust_remote_code=True).to('cuda').eval()

x = {k: v.to('cuda:0') for k, v in x.items() if isinstance(v, torch.Tensor)}
x['pixel_values'] = x['pixel_values'].to(torch.bfloat16)
# x['pixel_values'] = load_image('sana.png', max_num=12).to(torch.bfloat16).cuda()
# print(x)
with torch.no_grad():
    output = model.generate(**x, max_new_tokens=200, do_sample=True)
    print(processor1o.batch_decode(output, skip_special_tokens=False)[0])