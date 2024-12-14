
import numpy as np
import torch
import torch.nn as nn


class Embeddings(nn.Module):
    def __init__(self, vocab_size: int, emb_dims: int = 512) -> None:
        """Initializer for embedding block

        Args:
            vocab_size (int): Dimension of vocabulary as per the given
                tokenizer and dataset.
            emb_dims (int, optional): Dimensionality of the embedding. Defaults to 512.
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.emb_dims = emb_dims
        self.emb_scalar = torch.Tensor([np.sqrt(self.emb_dims)]).to(torch.float32)
        self.emb_layer = nn.Embedding(self.vocab_size, self.emb_dims)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input     : [batch, seq_len]
        # output    : [batch, seq_len, emb_size]

        embeddings = self.emb_layer(x) * self.emb_scalar.to(
            x.device
        )  # [batch, seq_len, emb_size]
        return embeddings
