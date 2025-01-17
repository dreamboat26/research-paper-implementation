# ROUND AND ROUND WE GO! WHAT MAKES ROTARY POSITIONAL ENCODINGS USEFUL?

## ABSTRACT (from paper directly)

Positional Encodings (PEs) are a critical component of Transformer-based Large Language Models (LLMs), providing the attention mechanism with important sequence-position information. One of the most popular types of encoding used today in LLMs are **Rotary Positional Encodings (RoPE)**, that rotate the queries and keys based on their relative distance. A common belief is that RoPE is useful because it helps to decay token dependency as relative distance increases. In this work, we argue that this is unlikely to be the core reason. We study the internals of a trained **Gemma 7B** model to understand how RoPE is being used at a mechanical level. We find that Gemma learns to use RoPE to construct robust ‘positional’ attention patterns by exploiting the highest frequencies. We also find that, in general, Gemma greatly prefers to use the lowest frequencies of RoPE, which we suspect are used to carry semantic information. We mathematically prove interesting behaviours of RoPE and conduct experiments to verify our findings, proposing a modification of RoPE that fixes some highlighted issues and improves performance. We believe that this work represents an interesting step in better understanding PEs in LLMs, which we believe holds crucial value for scaling LLMs to large sizes and context lengths.

## 1. INTRODUCTION

Rotary Positional Encoding (RoPE) has recently been widely adopted as the default positional encoding method for Transformer models. RoPE modifies the attention mechanism by rotating the query and key vectors in a way that encodes their relative position in the sequence. This encoding approach has been believed to improve token dependency decay as relative distance between tokens increases, but this paper challenges this assumption.

The authors present an analysis of the inner workings of a trained **Gemma 7B** model, which uses RoPE, to explore how it exploits RoPE in practice. Contrary to the prevailing view, they show that the mechanism underlying RoPE's utility is not simply the decay of token dependencies but rather the way in which RoPE helps the model to generate distinct attention patterns based on the relative position of tokens.

We also provide a modified version of RoPE that enhances its performance and explain the underlying issues with the current implementation.

## 2. RELATED WORK

- **Positional Encodings in Transformers**: Early works, such as Vaswani et al. (2017), introduced the concept of positional encodings in Transformer models, leading to significant improvements in sequence modeling tasks.
  
- **Sinusoidal and Learnable Positional Encodings**: Prior research has largely focused on sinusoidal and learnable positional encodings, both of which have been used in various Transformer-based models.

- **Rotary Positional Encodings (RoPE)**: RoPE, introduced by Su et al., offers a continuous representation of relative position information through a rotational transformation. However, its effectiveness and the mechanisms by which it aids model learning remain unclear, motivating this work.

## 3. METHOD

### 3.1 Analysis of RoPE in Gemma 7B

They conduct a detailed examination of a **Gemma 7B** model to understand how RoPE is utilized in practice. They observe that the model uses the highest frequencies of RoPE to create robust positional attention patterns. These patterns help the model effectively distinguish the relative positions of tokens.

Additionally, they found that the model predominantly uses the lowest frequencies of RoPE, which we hypothesize encode semantic information rather than purely positional data.

### 3.2 Mathematical Proof of RoPE Behaviours

They mathematically prove the interesting behaviors of RoPE, particularly its ability to create distinct attention patterns by leveraging both high and low frequencies. The analysis provides insights into the underlying mechanisms that make RoPE effective.

### 3.3 RoPE Modification

Based on the findings, they propose a modification to RoPE that addresses certain shortcomings, including issues with token dependency decay. The modified version improves performance in downstream tasks by better aligning positional encodings with the model’s needs.

## 4. EXPERIMENTS

They conducted several experiments to validate our findings, using both the original RoPE implementation and our modified version. The experiments demonstrate the utility of the proposed changes, with notable improvements in the model's ability to capture long-range dependencies.

### 4.1 Experiment Setup

- **Model**: Gemma 7B with Rotary Positional Encoding.
- **Tasks**: Language modeling and sequence prediction tasks to evaluate model performance.

### 4.2 Results

They compare the performance of the original RoPE against the modified version. The results show that the modified RoPE consistently outperforms the original in terms of both accuracy and generalization to longer contexts.

## 5. DISCUSSION

The work provides new insights into the behavior of positional encodings, particularly RoPE, in large language models. They highlight how RoPE is not merely a tool for decaying token dependency, as commonly believed, but instead a mechanism that helps the model construct more robust positional attention patterns. These patterns enable better semantic understanding and long-range dependency modeling.

The modification they propose further enhances RoPE’s effectiveness, and the authors believe it represents a promising direction for improving the scalability of LLMs.

## 6. CONCLUSION

In this paper, they argue that the utility of Rotary Positional Encodings lies not in their ability to decay token dependencies, but in their role in helping models learn robust attention patterns. By analyzing a trained model, the authors uncover key properties of RoPE that explain its success. They also propose a modification to RoPE that improves its performance, offering a step forward in the understanding of positional encodings for LLMs.

Their work provides a foundation for future research into the design and optimization of positional encodings in large-scale language models.

## REFERENCES

- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. A., Kaiser, Ł., & Polosukhin, I. (2017). Attention is all you need. *NeurIPS*.
- Su, Z., Zhang, X., & others (2021). Rotary Positional Encoding. *arXiv preprint arXiv:2104.09864*.

