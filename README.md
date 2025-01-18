## **Learning to (Learn at Test Time): RNNs with Expressive Hidden States**

### Problem:
- Traditional RNNs struggle with **generalizing** to unseen tasks or adapting to new patterns during inference, especially in real-world scenarios.

### Solution:
- This paper introduces **expressive hidden states** in RNNs that dynamically adjust during **test time** to improve task generalization and adaptability.

### Architecture:
- **Expressive Hidden States**: The hidden state is made **more adaptive** by introducing **learnable parameters** that allow it to **adjust dynamically** based on the task it encounters. The hidden state is made more expressive by allowing it to evolve through learnable parameters. Instead of just updating the hidden state based on the previous state and current input (as in traditional RNNs), the parameters that define the hidden state transformation can be modified to accommodate different tasks. Essentially, the network learns how to modify the structure of the hidden state, which allows it to better capture the complexity of the task-specific data. The paper proposes a parameterized transformation of the hidden state, where the update function (which normally updates the hidden state based on the current input and previous hidden state) is augmented with learnable parameters that allow for task-specific adjustments. These learnable parameters can be learned during training in a way that they reflect the structure and dependencies of the task at hand. The network learns to adjust how information flows into the hidden state, giving it the ability to retain more nuanced features of the data.
- **Task-Specific Adaptation**: The model learns a **meta-optimization** strategy for adjusting the hidden state during test time, which enables it to handle **unseen tasks** and **domain shifts** effectively. The key innovation is that meta-optimization is used to allow the model to adjust its hidden state at test time. Instead of relying on the same hidden state across different tasks or sequences, the model learns a strategy for adapting the hidden state based on the characteristics of the task it is given at test time.
- **Fast Adaptation**: The model can quickly **adapt** to new tasks without needing to retrain, leveraging the expressiveness of the hidden states.

### Key Contributions:
- **Adaptation at Test Time**: Expressive hidden states allow the RNN to dynamically adapt to unseen data at test time, improving generalization.
- **Meta-Learning Approach**: The model learns a meta-optimization process to adjust its hidden states, enabling fast adaptation to new tasks without retraining.
