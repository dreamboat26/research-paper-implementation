## **B-STAR: Monitoring and Balancing Exploration and Exploitation in Self-Taught Reasoners**

### Paper Information
- **Authors:** Weihao Zeng, Yuzhen Huang, Lulu Zhao, Yijun Wang, Zifei Shan, Junxian He
- **Institutions:** Hong Kong University of Science and Technology, BAAI, Tencent
- **Published:** December 23, 2024
- **Link:** [GitHub Repository](https://github.com/hkust-nlp/B-STaR)

### Summary
**B-STAR** introduces a framework to balance exploration (diversity in solutions) and exploitation (selection of high-quality solutions) during iterative self-improvement processes in LLMs.

### Key Features
- **Dynamic Adjustments:** Adapts sampling temperature and reward thresholds based on a new balance score metric.
- **Exploration vs. Exploitation Monitoring:** Tracks metrics like Pass@K and Best-of-K accuracy to optimize training.
- **Iterative Refinement:** Continuously updates the model with high-quality, diverse solutions.

### Implementation
- **Self-Improvement Loop:** Combines solution generation, external reward evaluation, and model updating.
- **Dynamic Configuration:** Adjusts hyperparameters dynamically to maintain exploration and exploitation balance.

### Results
- Achieved superior performance on GSM8k, MATH, and other reasoning datasets.
- Retained exploration capabilities throughout training, avoiding stagnation observed in previous methods.

---

## How to Cite
If you use any of the methods described here, please cite the respective papers:

### B-STAR
```bibtex
@article{zeng2024bstar,
  title={B-STAR: Monitoring and Balancing Exploration and Exploitation in Self-Taught Reasoners},
  author={Zeng, Weihao and Huang, Yuzhen and others},
  journal={arXiv preprint arXiv:2412.17256},
  year={2024}
}
```
