##
# @author Meet Patel <>
# @file Description
# @desc Created on 2023-03-24 11:26:30 pm
# @copyright MIT License
#

import torch
import torch.nn as nn


class PositionalEmbeddings(nn.Module):
    def __init__(
        self, d_model: int = 512, drop_prob: float = 0.1, max_len: int = 5000
    ) -> None:
        """Initializer for Positional embedding block

        Args:
            d_model (int, optional): Dimension for hidden state.
                Defaults to 512.
            drop_prob (float, optional): Dropout rate to be used.
                Defaults to 0.1.
            max_len (int, optional): Theoretically possible max sequence length
                for any input. The higher it is the higher it the size of embedding
                array. Defaults to 5000.
        """
        super().__init__()

        self.d_model = d_model  # Embedding dimension
        self.dropout_layer = nn.Dropout(p=drop_prob)

        pe = torch.zeros(size=[max_len, d_model])

        positions = torch.arange(0, max_len).unsqueeze(1)  # [seq_len, 1]
        denominator = torch.pow(
            10000, torch.arange(0, self.d_model, 2) / self.d_model
        )  # [emb_size // 2]
        pe[:, 0::2] = torch.sin(positions / denominator)  # [seq_len, emb_size // 2]
        pe[:, 1::2] = torch.cos(positions / denominator)  # [seq_len, emb_size // 2]
        pe = pe.unsqueeze(0)  # [batch, seq_len, emb_size]

        # Registering so as to pass this tensor to GPU if req. we calling model.cuda()
        # But using persistent as False as this is a large tensor and we shall not transfer
        # it to GPU.
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input     : [batch, seq_len, emb_size]
        # output    : [batch, seq_len, emb_size]

        x = x + self.pe[:, : x.shape[1], :]
        x = self.dropout_layer(x)

        return x
