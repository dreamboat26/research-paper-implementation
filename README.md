# Differential Attention Mechanism

In the Differential Transformer model, the differential attention mechanism is designed to map query, key, and value vectors to outputs while canceling out the noise of attention scores. The key innovation lies in using two softmax functions to eliminate the noise, improving the attention mechanism's performance.

## Multi-Head Differential Attention

The multi-head differential attention mechanism operates by applying the differential attention operator across multiple attention heads. Each head computes the difference between two softmax attention maps to cancel out attention noise. The results from all heads are then concatenated and processed further.

### Code Implementation:

```python
import torch
import torch.nn.functional as F
from math import sqrt

def DiffAttn(X, W_q, W_k, W_v, λ):
    # Project X to queries, keys, and values
    Q1, Q2 = torch.split(X @ W_q, split_size_or_sections=X.shape[-1]//2, dim=-1)
    K1, K2 = torch.split(X @ W_k, split_size_or_sections=X.shape[-1]//2, dim=-1)
    V = X @ W_v

    # Compute the attention scores
    s = 1 / sqrt(X.shape[-1])
    A1 = Q1 @ K1.transpose(-1, -2) * s
    A2 = Q2 @ K2.transpose(-1, -2) * s

    # Apply the differential attention formula
    return (F.softmax(A1, dim=-1) - λ * F.softmax(A2, dim=-1)) @ V

def MultiHead(X, W_q, W_k, W_v, W_o, λ, h, λinit):
    # Compute multi-head attention
    heads = [DiffAttn(X, W_q[i], W_k[i], W_v[i], λ) for i in range(h)]
    
    # Apply GroupNorm to each head
    O = torch.stack(heads, dim=1)
    O = F.group_norm(O, num_groups=h)
    
    # Apply the fixed multiplier (1 - λinit) and concatenate
    O = O * (1 - λinit)
    
    return O.flatten(1) @ W_o
```

## Diagram of Multi-Head Differential Attention:
      X  ---> [Linear] ---> Q1, Q2
                 |
            [Softmax] ---> Attention Scores
                 |
            [Linear] ---> K1, K2
                 |
            [Softmax] ---> Attention Scores
                 |
      [Q1; Q2] --- (Cancel Noise) ---> V ---> Output
In this diagram, the key steps are the projections of the input XX to query, key, and value vectors, followed by the calculation of attention scores using the two softmax functions. The noise-canceling effect is achieved by subtracting the second softmax function's result from the first.

## Insights
This design is inspired by differential amplifiers used in electrical engineering, where the difference between two signals is used to cancel out noise. Similarly, in noise-canceling headphones, the external noise is subtracted from the desired signal, leading to cleaner audio. This approach significantly improves the efficiency and accuracy of the attention mechanism in the Transformer model.
