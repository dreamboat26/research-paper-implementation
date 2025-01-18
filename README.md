## **Learning to (Learn at Test Time): RNNs with Expressive Hidden States**

### Problem:
- Traditional RNNs struggle with **generalizing** to unseen tasks or adapting to new patterns during inference, especially in real-world scenarios.

### Solution:
- This paper introduces **expressive hidden states** in RNNs that dynamically adjust during **test time** to improve task generalization and adaptability.

### Architecture:
- **Expressive Hidden States**: The hidden state is made **more adaptive** by introducing **learnable parameters** that allow it to **adjust dynamically** based on the task it encounters.
- **Task-Specific Adaptation**: The model learns a **meta-optimization** strategy for adjusting the hidden state during test time, which enables it to handle **unseen tasks** and **domain shifts** effectively.
- **Fast Adaptation**: The model can quickly **adapt** to new tasks without needing to retrain, leveraging the expressiveness of the hidden states.

### Key Contributions:
- **Adaptation at Test Time**: Expressive hidden states allow the RNN to dynamically adapt to unseen data at test time, improving generalization.
- **Meta-Learning Approach**: The model learns a meta-optimization process to adjust its hidden states, enabling fast adaptation to new tasks without retraining.
