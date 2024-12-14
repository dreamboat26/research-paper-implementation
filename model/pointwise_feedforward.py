
import torch
import torch.nn as nn


class PointwiseFeedForwardNetwork(nn.Module):
    def __init__(self, d_model: int = 512, d_ff: int = 2048, drop_prob: float = 0.1):
        """Initializer for Feedforward layers block inside single encoder / decoder
        layer.

        Args:
            d_model (int, optional): Dimension for hidden states in encoder.
                Defaults to 512.
            d_ff (int, optional): Dimensionality for hidden layer of Feedforward
                layer. Defaults to 2048.
            drop_prob (float, optional): Dropout rate to be used. Defaults to 0.1.
        """
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout_layer = nn.Dropout(drop_prob)
        self.linear_1 = nn.Linear(self.d_model, self.d_ff, bias=True)
        self.linear_2 = nn.Linear(self.d_ff, self.d_model, bias=True)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input     : [batch, seq_len, emb_size]
        # output    : [batch, seq_len, emb_size]

        x = self.linear_2(self.dropout_layer(self.relu(self.linear_1(x))))
        return x
