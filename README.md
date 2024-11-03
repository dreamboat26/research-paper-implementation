# SVDiff
SVDiff is a lightweight method for finetuning models text-to-image models on new objects, styles etc. In short, this method simply applies the SVD decomposition on weight matrices and only finetunes the V matrix, which technically is only a vector. This results in checkpoints being extremely small (e.g. 4MB for SDXL when finetuning the UNet & both text-encoders). You can read more about the method here in [the paper](https://arxiv.org/abs/2303.11305).
### Example
<br>
<img src="https://github.com/dome272/SVDiff/assets/61938694/1ae1fd17-5796-4ee4-b659-a95e367d7ed2" width="300" height="300">

The method can achieve results like this after ~1000 steps:
<img src="https://github.com/dreamboat26/research-paper-implementation/blob/SVDiff/1.png">

### Supported Models
- [x] Stable Diffusion XL
- [ ] [Würstchen](https://github.com/dome272/Wuerstchen/)
- [ ] Stable Diffusion 1.4

### Run Finetuning
For now you can just look into the provided jupyter notebook. Adapt & run the cells. 
