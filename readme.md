# Paper Abstract
Conditional text-to-image generation has seen countless recent improvements in terms of quality, diversity and fidelity. Nevertheless, most state-of-the-art models require numerous inference steps to produce faithful generations, resulting in performance bottlenecks for end-user applications. In this paper we introduce Paella, a novel text-to-image model requiring less than 10 steps to sample high-fidelity images, using a speed-optimized architecture allowing to sample a single image in less than 500 ms, while having 573M parameters. The model operates on a compressed & quantized latent space, it is conditioned on CLIP embeddings and uses an improved sampling function over previous works. Aside from text-conditional image generation, our model is able to do latent space interpolation and image manipulations such as inpainting, outpainting, and structural editing.
<br>
<br>
![collage](https://user-images.githubusercontent.com/61938694/231021615-38df0a0a-d97e-4f7a-99d9-99952357b4b1.png)

Please find all details about the model and how it was trained in their paper [paper on arxiv](https://arxiv.org/abs/2211.07292v2).
<hr>

## Code
I especially want to highlight the minimalistic amount of code that is necessary to run & train Paella. The training & sampling code can fit in under 140 lines of code. 

### License
The model code and weights are released under the [MIT license].
