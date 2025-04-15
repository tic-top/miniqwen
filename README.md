# A trial to reimplement 4o

## 2 Stage training
### Stage 1 pretraining
Open sources data: pix2pix, kosmosg里面应该有，GOT里面有
t2i
ti2ti
ti2ti2ti...
kosmosg: LAION-2B [SBV+22], LAION-400M [SVB+21], COYO-700M [BPK+22], and Conceptual Captions [SDGS18, CSDS21]



### Stage instruction tunning
1. Single- and multi-turn language-guided image-editing from MagicBrush (Zhang et al., 2024) and
SEED (Ge et al., 2024) (only involving the real-world and multi-turn subsets).
2. Dense prediction tasks, including surface norm estimation from NYUv2 Silberman et al. (2012)
and ScanNet Dai et al. (2017), depth estimation from Kitti v2 Cabon et al. (2020) and Sintel Butler
et al. (2012), pose estimation from MSCOCO Lin et al. (2014), semantic segmentation data annotated
with OneFormer (Jain et al., 2023) on image from Laion (Schuhmann et al., 2022), and grounding
data from RefCOCO (Kazemzadeh et al., 2014).


Metaquery提到的构造的数据集
还是多个图片作为input，但是output只是一张图片




<image>
<image>

<gen_image>

<gen_image>

<image>


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




todo 

sana pipelin xformers 12.1TypeError: SanaTransformer2DModel.__init__() got an unexpected keyword argument 'torch_dtype'