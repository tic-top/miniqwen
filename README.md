# two state

magician-blue/mini4o

## Input illustration
```
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
            },
            {"type": "text", "text": "Describe this image.\n"},
            {
                "type": "image_gen",
                "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
            },
            {
                "type": "image",
                "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
            },
            {
                "type": "image_gen",
                "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
            },
        ],
    }
]

### after apply chat template
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
<|vision_start|><|image_pad|><|vision_end|>
Describe this image.<|image_gen_start|><|image_gen_pad|><|image_gen_end|>
<|vision_start|><|image_pad|><|vision_end|>
<|image_gen_start|><|image_gen_pad|><|image_gen_end|>
<|im_end|>
<|im_start|>assistant


### after the processor
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
<|vision_start|><|image_pad|>*n<|vision_end|>
Describe this image.
<|image_gen_start|><|image_gen_pad|>*256<|image_gen_end|>
<|vision_start|><|image_pad|><|vision_end|>
<|image_gen_start|><|image_gen_pad|>*256<|image_gen_end|>
<|im_end|>
<|im_start|>assistant
```


## stage 1 pretraining to align the image
可能没有图文交错的生成任务

## stage 2 instruction tunning
还是多个图片作为input，但是output只是一张图片

## 还是可以构造一个交错的，一会儿理解一会儿生成的数据集

<image>
<image>

<gen_image>

<gen_image>

<image>

任意交错的才是最好的
