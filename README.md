# Paper Summary

## 4. Cached Transformers: Improving Transformers with Differentiable Memory Cache
### Overview
**Cached Transformers** introduce a method for efficiently storing and accessing previously computed activations using a **differentiable memory cache**. This allows transformers to process long sequences without the need to recompute activations for the same tokens multiple times, making the model more efficient and scalable for long-sequence tasks.

### Key Contributions
- **Memory Cache Mechanism**: Introduces a differentiable memory cache that stores previously computed activations, allowing the transformer to retrieve and reuse them during later computations.
- **Efficient Sequence Processing**: By caching and reusing activations, the model can process longer sequences without significant increases in computational cost.
- **Differentiable Memory**: The memory cache is differentiable, which means it can be updated during backpropagation, making it compatible with end-to-end training.

### Architecture
- **Memory Cache Integration**: Each layer of the transformer has access to a memory cache that stores activations of previously processed tokens.
- **Cache Retrieval Mechanism**: The model learns when to retrieve from the cache and when to recompute activations, improving the efficiency of processing long sequences.
- **End-to-End Differentiability**: The memory cache is incorporated into the model in a way that allows gradient flow through the cache, ensuring that the model can be trained in the usual manner.

### Applications
- **Long Sequence Processing**: Allows transformers to handle longer sequences without running into memory or computational bottlenecks.
- **Efficient NLP**: Useful in tasks like document classification, machine translation, and any NLP task that involves long-range dependencies.

---

