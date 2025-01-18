# Summary of Paper

## **Infini-attention: Infinite Context Transformers**

### Problem:
- **Fixed Context Window**: Standard transformer models have a **fixed context window**, limiting the amount of input data that can be processed at once (e.g., 2048 tokens). This is problematic for tasks that require **long-range dependencies**.

### Solution:
- **Infinite Context Attention**: **Infini-attention** introduces a **scalable attention mechanism** that allows transformers to process **arbitrary-length sequences** without exponentially increasing memory usage.

### Architecture:
- **Hierarchical Attention**: Breaks long sequences into smaller, manageable chunks and allows each chunk to attend to others at different levels of the hierarchy. This reduces the need for an attention matrix for every token.
- **Memory Efficiency**: Instead of recalculating attention for all tokens, the model stores only the most relevant information and attends to it dynamically, which makes it scalable for long sequences.
- **Adaptive Attention**: The attention span adapts based on the sequence length and relevance, dynamically adjusting which parts of the sequence are attended to.

### Key Contribution:
- **Infinite Context Attention**: Allows transformers to process **sequences of arbitrary length** while maintaining memory and computational efficiency, making it suitable for long-range dependency tasks like document summarization.
---

## Summary of Key Contributions:

- **Infini-attention**: Provides an **infinite context attention mechanism** that enables transformers to handle long sequences without overwhelming memory, 

---

## Conclusion

- **Infini-attention** solves the fixed context window limitation in transformers.
