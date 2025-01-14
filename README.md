## **Training Large Language Models to Reason in a Continuous Latent Space (Coconut)**

### Paper Information
- **Authors:** Shibo Hao, Sainbayar Sukhbaatar, DiJia Su, Xian Li, Zhiting Hu, Jason Weston, Yuandong Tian
- **Institutions:** FAIR at Meta, UC San Diego
- **Published:** December 12, 2024
- **Link:** [ArXiv](https://arxiv.org/abs/2412.06769)

### Summary
This paper introduces **Coconut** (Chain of Continuous Thought), a novel paradigm for reasoning in a continuous latent space instead of the traditional language space. By feeding the hidden state of one reasoning step as the input for the next, Coconut allows for efficient and flexible reasoning without generating intermediate language tokens.

### Key Features
- **Latent Reasoning Mode:** Uses hidden states as representations of reasoning steps.
- **Switching Mechanism:** Controlled by `<bot>` (beginning of thought) and `<eot>` (end of thought) tokens.
- **Multi-Stage Curriculum Training:** Gradual replacement of language reasoning steps with latent thoughts.
- **Breadth-First Search-Like Reasoning:** Encodes multiple potential reasoning paths simultaneously.

### Implementation
- **Training:** Combines traditional CoT data with continuous latent space representations, supervised by a multi-stage curriculum.
- **Inference:** Alternates between latent and language modes based on special tokens.

### Results
- Outperforms traditional CoT on tasks like GSM8k (math reasoning) and ProntoQA (logical reasoning).
- Requires fewer tokens during inference, making it computationally efficient.

---

## How to Cite
If you use any of the methods described here, please cite the respective paper:

### Coconut
```bibtex
@article{hao2024coconut,
  title={Training Large Language Models to Reason in a Continuous Latent Space},
  author={Hao, Shibo and Sukhbaatar, Sainbayar and others},
  journal={arXiv preprint arXiv:2412.06769},
  year={2024}
}
```

Feel free to explore each folder for specific implementations and additional insights!

