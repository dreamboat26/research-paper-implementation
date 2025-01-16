# Byte Latent Transformer (BLT)

### Introduction
The **Byte Latent Transformer (BLT)** is a tokenizer-free large language model (LLM) architecture that directly processes raw byte-level data, eliminating the need for tokenization. BLT achieves state-of-the-art performance while improving inference efficiency and robustness.

Key Highlights:
- **Dynamic Patching:** Groups bytes into variable-sized patches based on entropy.
- **Efficiency:** Matches or outperforms tokenization-based models like LLaMA while using fewer computational resources.
- **Robustness:** Superior performance on noisy input and character-level tasks.

> **Paper:** [Byte Latent Transformer: Patches Scale Better Than Tokens](https://arxiv.org/abs/2412.09871)
>  
> **Authors:** Artidoro Pagnoni, Ram Pasunuru, Pedro Rodriguez, et al.

---

### Architecture
BLT consists of three main components:

1. **Local Encoder**
   - Dynamically groups bytes into patches using entropy-based patching.
   - Utilizes lightweight transformer layers and hash n-gram embeddings for efficient byte-level representation.

2. **Latent Global Transformer**
   - Processes patch representations instead of individual bytes.
   - Employs block-causal attention and scales efficiently with model size and patch size.

3. **Local Decoder**
   - Reconstructs byte sequences from patch-level representations.
   - Uses cross-attention to map patches back to bytes.

![BLT Architecture](https://github.com/facebookresearch/blt/blob/main/images/blt_architecture.png)

---

### Key Features
- **Dynamic Entropy-Based Patching:** Adapts patch size based on input complexity.
- **Flop Efficiency:** Saves up to 50% inference FLOPs compared to tokenization-based models.
- **Robust to Noise:** Enhanced performance on noisy and multilingual tasks.
- **Scalability:** Achieves better scaling trends by increasing patch and model size simultaneously.

---

### Results
1. **Efficiency**
   - BLT significantly reduces inference FLOPs while maintaining performance parity with tokenization-based models.

2. **Performance on Benchmarks**
   - **HellaSwag (0-shot):** BLT: 80.6% vs LLaMA 3: 79.1%.
   - **MMLU (5-shot):** BLT: 57.4% vs LLaMA 3: 58.1%.

3. **Robustness**
   - Outperforms tokenization-based models on noisy inputs and character-level benchmarks.

---

### Installation
To use BLT, clone the repository and install dependencies:

```bash
# Clone the repository
git clone https://github.com/facebookresearch/blt.git
cd blt

# Install dependencies
pip install -r requirements.txt
```

---

### Usage
#### Training BLT
BLT models can be trained using the scripts provided in the repository. Example command:

```bash
python train_blt.py \
    --dataset_path <dataset_path> \
    --output_dir <output_dir> \
    --model_size 8B \
    --patching_method entropy \
    --flop_budget <flops>
```

#### Inference
Run inference using a pre-trained BLT model:

```bash
python infer_blt.py \
    --model_path <model_checkpoint> \
    --input_text "Your input here"
```

---

### Dataset
BLT was trained on:
1. **LLaMA 2 Dataset:** 2 trillion tokens from diverse sources.
2. **BLT-1T Dataset:** 1 trillion tokens curated for high quality.

---

### Citation
If you use BLT in your research, please cite the paper:

```bibtex
@article{pagnoni2024blt,
  title={Byte Latent Transformer: Patches Scale Better Than Tokens},
  author={Artidoro Pagnoni and Ram Pasunuru and Pedro Rodriguez and others},
  journal={arXiv preprint arXiv:2412.09871},
  year={2024}
}
```

---

### License
This project is licensed under the [MIT License](LICENSE).

---

### Acknowledgements
This work was conducted by FAIR at Meta, with contributions from researchers at the University of Washington and the University of Chicago.


