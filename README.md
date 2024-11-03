[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1KeN407dItcjLcWdMLrByZ8mPa1MT2_DJ?usp=sharing)

# Würstchen (ICLR 2024)

![huggingface-blog-post-thumbnail](https://github.com/dreamboat26/research-paper-implementation/blob/Wuerstchen/1.jpg)

## What is this?
Würstchen is a new framework for training text-conditional models by moving the computationally expensive text-conditional stage into a highly compressed latent space. Common approaches make use of a single stage compression, while Würstchen introduces another Stage that introduces even more compression. In total we have Stage A & B that are responsible for compressing images and Stage C that learns the text-conditional part in the low dimensional latent space. With that Würstchen achieves a 42x compression factor, while still reconstructing images faithfully. This enables training of Stage C to be fast and computationally cheap. We refer to [the paper](https://arxiv.org/abs/2306.00637) for details.

### Using in 🧨 diffusers

Würstchen is fully integrated into the [`diffusers` library](https://huggingface.co/docs/diffusers). Here's how to use it: 

```python
# pip install -U transformers accelerate diffusers

import torch
from diffusers import AutoPipelineForText2Image
from diffusers.pipelines.wuerstchen import DEFAULT_STAGE_C_TIMESTEPS

pipe = AutoPipelineForText2Image.from_pretrained("warp-ai/wuerstchen", torch_dtype=torch.float16).to("cuda")

caption = "Anthropomorphic cat dressed as a fire fighter"
images = pipe(
    caption, 
    width=1024,
    height=1536,
    prior_timesteps=DEFAULT_STAGE_C_TIMESTEPS,
    prior_guidance_scale=4.0,
    num_images_per_prompt=2,
).images
```
Refer to the [official documentation](https://huggingface.co/docs/diffusers/main/en/api/pipelines/wuerstchen) to learn more. 

## Citation
If you use our approach in your research or were inspired by it, we would be thrilled if you cite our paper:
```
@inproceedings{
            pernias2024wrstchen,
            title={W\"urstchen: An Efficient Architecture for Large-Scale Text-to-Image Diffusion Models},
            author={Pablo Pernias and Dominic Rampas and Mats Leon Richter and Christopher Pal and Marc Aubreville},
            booktitle={The Twelfth International Conference on Learning Representations},
            year={2024},
            url={https://openreview.net/forum?id=gU58d5QeGv}
      }
```
