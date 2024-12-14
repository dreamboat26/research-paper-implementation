##
# @author Meet Patel <>
# @file Description
# @desc Created on 2023-04-08 7:31:48 pm
# @copyright MIT License
#

import torch
import torch.nn as nn

from model.attention import MultiHeadAttention
from model.pointwise_feedforward import PointwiseFeedForwardNetwork


class EncoderBlock(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        d_k: int = 64,
        d_v: int = 64,
        d_ff: int = 2048,
        num_heads: int = 8,
        bias: bool = True,
        drop_prob: float = 0.1,
        pad_token: int = 0,
    ) -> None:
        """Initializer for single encoder block

        Args:
            d_model (int, optional): Dimension for hidden states in encoder.
                Defaults to 512.
            d_k (int, optional): Dimensionality for Keys. Defaults to 64.
            d_v (int, optional): Dimensionality for Query. Defaults to 64.
            d_ff (int, optional): Dimensionality for hidden layer of Feedforward
                layer. Defaults to 2048.
            num_heads (int, optional): Number of parallel heads in Multi head
                attention block. Defaults to 8.
            bias (bool, optional): Use bias for all the layers. Defaults to True.
            drop_prob (float, optional): Dropout rate used across encoder.
                Defaults to 0.1.
            pad_token (int, optional): Pad token to be used for masked softmax.
                Defaults to 0.
        """
        super().__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.drop_prob = drop_prob
        self.num_heads = num_heads
        self.bias = bias
        self.d_ff = d_ff
        self.pad_token = pad_token
        self.mha = MultiHeadAttention(
            self.d_model,
            self.d_k,
            self.d_v,
            self.drop_prob,
            self.num_heads,
            self.bias,
            self.pad_token,
        )
        self.pff = PointwiseFeedForwardNetwork(self.d_model, self.d_ff, self.drop_prob)
        self.dropout_layer_1 = nn.Dropout(drop_prob)
        self.dropout_layer_2 = nn.Dropout(drop_prob)
        self.layer_norm_1 = nn.LayerNorm(self.d_model)
        self.layer_norm_2 = nn.LayerNorm(self.d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # input     : [batch, seq_len, emb_size]
        # mask      : [batch, 1, 1, seq_len]
        # output    : [batch, seq_len, emb_size]

        mha_res = self.mha(x, x, x, mask)  # [batch, seq_len, emb_size]
        mha_res = self.dropout_layer_1(mha_res)
        mha_res = self.layer_norm_1(x + mha_res)  # [batch, seq_len, emb_size]

        pff_res = self.pff(mha_res)  # [batch, seq_len, emb_size]
        pff_res = self.dropout_layer_2(pff_res)
        pff_res = self.layer_norm_2(mha_res + pff_res)  # [batch, seq_len, emb_size]
        return pff_res


class Encoder(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        d_k: int = 64,
        d_v: int = 64,
        d_ff: int = 2048,
        num_heads: int = 8,
        n_blocks: int = 6,
        use_bias: bool = True,
        dropout: float = 0.1,
        pad_token: int = 0,
    ):
        """Initializer for encoder class.

        Args:
            d_model (int, optional): Dimension for hidden states in encoder.
                Defaults to 512.
            d_k (int, optional): Dimensionality for Keys. Defaults to 64.
            d_v (int, optional): Dimensionality for Query. Defaults to 64.
            d_ff (int, optional): Dimensionality for hidden layer of Feedforward
                layer. Defaults to 2048.
            num_heads (int, optional): Number of parallel heads in Multi head
                attention block. Defaults to 8.
            n_blocks (int, optional): Number of encoder blocks in Encoder part
                of the Transformers. Defaults to 6.
            use_bias (bool, optional): Use bias for all the layers. Defaults to True.
            dropout (float, optional): Dropout rate used across encoder.
                Defaults to 0.1.
            pad_token (int, optional): Pad token to be used for masked softmax.
                Defaults to 0.
        """
        super().__init__()
        self.encoders = nn.ModuleList(
            [
                EncoderBlock(
                    d_model, d_k, d_v, d_ff, num_heads, use_bias, dropout, pad_token
                )
                for _ in range(n_blocks)
            ]
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # input     : [batch, seq_len, emb_size]
        # mask      : [batch, 1, 1, seq_len] for self attention in encoder
        # output    : [batch, seq_len, emb_size]

        for encoder in self.encoders:
            x = encoder(x, mask)
        return x
