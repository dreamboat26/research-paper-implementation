
import torch
import torch.nn as nn

from model.attention import MultiHeadAttention
from model.pointwise_feedforward import PointwiseFeedForwardNetwork


class DecoderBlock(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        d_k: int = 64,
        d_v: int = 64,
        d_ff: int = 2048,
        num_heads: int = 8,
        use_bias: bool = True,
        drop_prob: float = 0.1,
        pad_token: int = 0,
    ):
        """Initializer for single decoder block.

        Args:
            d_model (int, optional): Dimension for hidden states in decoder.
                Defaults to 512.
            d_k (int, optional): Dimensionality for Keys. Defaults to 64.
            d_v (int, optional): Dimensionality for Query. Defaults to 64.
            d_ff (int, optional): Dimensionality for hidden layer of Feedforward
                layer. Defaults to 2048.
            num_heads (int, optional): Number of parallel heads in Multi head
                attention block. Defaults to 8.
            use_bias (bool, optional): Use bias for all the layers. Defaults to True.
            dropout (float, optional): Dropout rate used across decoder.
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
        self.bias = use_bias
        self.d_ff = d_ff
        self.pad_token = pad_token
        # Masked Multi Head Attention
        self.mmha = MultiHeadAttention(
            self.d_model,
            self.d_k,
            self.d_v,
            self.drop_prob,
            self.num_heads,
            self.bias,
            self.pad_token,
        )
        # Multi Head Cross Attention
        self.mhca = MultiHeadAttention(
            self.d_model,
            self.d_k,
            self.d_v,
            self.drop_prob,
            self.num_heads,
            self.bias,
            self.pad_token,
        )
        # Pointwise Feedforward Network
        self.pff = PointwiseFeedForwardNetwork(self.d_model, self.d_ff, self.drop_prob)

        self.dropout_layer = nn.Dropout(self.drop_prob)
        self.layer_norm_1 = nn.LayerNorm(self.d_model)
        self.layer_norm_2 = nn.LayerNorm(self.d_model)
        self.layer_norm_3 = nn.LayerNorm(self.d_model)

    def forward(
        self,
        enc_rep_orig: torch.Tensor,
        dec_rep_orig: torch.Tensor,
        enc_mask: torch.Tensor,
        dec_mask: torch.Tensor,
    ) -> torch.Tensor:
        # enc_rep_orig  : [batch, seq_len, emb_size]
        # dec_rep_orig  : [batch, seq_len, emb_size]
        # enc_mask      : [batch, 1, 1, seq_len] for self attention
        # dec_mask      : [batch, 1, seq_len, seq_len] for cross attention
        # output        : [batch, seq_len, emb_size]

        dec_rep_1 = self.mmha(
            dec_rep_orig, dec_rep_orig, dec_rep_orig, dec_mask
        )  # [batch, seq_len, emb_size]
        dec_rep_1 = self.dropout_layer(dec_rep_1)
        dec_rep_1 = self.layer_norm_1(dec_rep_orig + dec_rep_1)

        dec_rep_2 = self.mhca(
            dec_rep_1, enc_rep_orig, enc_rep_orig, enc_mask
        )  # [batch, seq_len, emb_size]
        dec_rep_2 = self.dropout_layer(dec_rep_2)
        dec_rep_2 = self.layer_norm_1(dec_rep_1 + dec_rep_2)

        dec_rep_3 = self.pff(dec_rep_2)  # [batch, seq_len, emb_size]
        dec_rep_3 = self.dropout_layer(dec_rep_3)
        dec_rep_3 = self.layer_norm_1(dec_rep_2 + dec_rep_3)
        return dec_rep_3


class Decoder(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        d_k: int = 64,
        d_v: int = 64,
        d_ff: int = 2048,
        num_heads: int = 8,
        n_blocks: int = 6,
        use_bias: bool = True,
        drop_prob: float = 0.1,
        pad_token: int = 0,
    ) -> None:
        """Initializer for decoder class.

        Args:
            d_model (int, optional): Dimension for hidden states in decoder.
                Defaults to 512.
            d_k (int, optional): Dimensionality for Keys. Defaults to 64.
            d_v (int, optional): Dimensionality for Query. Defaults to 64.
            d_ff (int, optional): Dimensionality for hidden layer of Feedforward
                layer. Defaults to 2048.
            num_heads (int, optional): Number of parallel heads in Multi head
                attention block. Defaults to 8.
            n_blocks (int, optional): Number of decoder blocks in Decoder part
                of the Transformers. Defaults to 6.
            use_bias (bool, optional): Use bias for all the layers. Defaults to True.
            dropout (float, optional): Dropout rate used across decoder.
                Defaults to 0.1.
            pad_token (int, optional): Pad token to be used for masked softmax.
                Defaults to 0.
        """
        super().__init__()
        self.decoders = nn.ModuleList(
            [
                DecoderBlock(
                    d_model, d_k, d_v, d_ff, num_heads, use_bias, drop_prob, pad_token
                )
                for _ in range(n_blocks)
            ]
        )

    def forward(
        self,
        enc_rep: torch.Tensor,
        dec_rep: torch.Tensor,
        enc_mask: torch.Tensor,
        dec_mask: torch.Tensor,
    ) -> torch.Tensor:
        # enc_rep_orig  : [batch, seq_len, emb_size]
        # dec_rep_orig  : [batch, seq_len, emb_size]
        # enc_mask      : [batch, 1, 1, seq_len] for encoder
        # dec_mask      : [batch, 1, seq_len, seq_len] for decoder
        # output        : [batch, seq_len, emb_size]

        output = dec_rep
        for decoder in self.decoders:
            output = decoder(enc_rep, output, enc_mask, dec_mask)
        return output


class DecoderGenerator(nn.Module):
    def __init__(
        self, vocab_size: int, d_model: int = 512, apply_softmax: bool = False
    ):
        """Initializer for decoder generator which converts the hidden representation
        into probabilities.

        Args:
            vocab_size (int): Decoder vocabulary size.
            d_model (int, optional): Dimension of hidden states in decoder.
                Defaults to 512.
            apply_softmax (bool, optional): Apply softmax on logits to return
                class probabilities. Defaults to False.
        """
        super().__init__()

        self.linear = nn.Linear(d_model, vocab_size)
        self.softmax = nn.Softmax(dim=-1)
        self.apply_softmax = apply_softmax

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input: [batch, seq_len, emb_size]
        # output: [batch, seq_len, vocab_size]

        x = self.linear(x)
        if self.apply_softmax:
            return self.softmax(x)
        return x
