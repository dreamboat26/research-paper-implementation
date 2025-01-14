## 3. **rStar: Mutual Reasoning Makes Smaller LLMs Stronger Problem-Solvers**

### Paper Information
- **Authors:** Zhenting Qi, Mingyuan Ma, Jiahang Xu, Li Lyna Zhang, Fan Yang, Mao Yang
- **Institutions:** Microsoft Research Asia, Harvard University
- **Published:** August 12, 2024
- **Link:** [GitHub Repository](https://github.com/zhentingqi/rStar)

### Summary
**rStar** improves the reasoning capabilities of smaller language models (SLMs) by leveraging self-play mutual reasoning. This method employs two SLMs: one generates reasoning trajectories, while the other verifies them.

### Key Features
- **Monte Carlo Tree Search (MCTS):** Augments reasoning with a diverse action space inspired by human problem-solving strategies.
- **Mutual Verification:** Uses a secondary SLM to validate generated reasoning trajectories.
- **Rich Action Space:** Includes actions like rephrasing questions, breaking down problems, and proposing sub-questions.

### Implementation
- **Generation Stage:** MCTS explores reasoning paths using a diverse action set.
- **Verification Stage:** A second SLM validates reasoning steps through mutual consistency.
- **Final Selection:** Trajectories with the highest rewards and consistency scores are chosen as solutions.

### Results
- Boosted reasoning accuracy significantly for datasets like GSM8k and MATH.
- Outperformed fine-tuned models without requiring superior teacher models.

---

## How to Cite
If you use any of the methods described here, please cite the respective paper:

### rStar
```bibtex
@article{qi2024rstar,
  title={Mutual Reasoning Makes Smaller LLMs Stronger Problem-Solvers},
  author={Qi, Zhenting and Ma, Mingyuan and others},
  journal={arXiv preprint arXiv:2408.06195},
  year={2024}
}
```

---

Feel free to explore each folder for specific implementations and additional insights!
