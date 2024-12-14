import torch
import torch.nn as nn

from model.decoder import Decoder, DecoderGenerator
from model.embeddings import Embeddings
from model.encoder import Encoder
from model.positional_embeddings import PositionalEmbeddings
from model.utils import get_src_pad_mask, get_trg_pad_mask


class TransformerEncoderDecoder(nn.Module):
    def __init__(
        self,
        enc_vocab_size,
        dec_vocab_size,
        d_model: int = 512,
        max_len: int = 3000,
        pos_dropout: float = 0.1,
        enc_dropout: float = 0.1,
        enc_num_heads: int = 8,
        enc_num_blocks: int = 6,
        enc_d_ff: int = 2048,
        enc_use_bias: bool = True,
        enc_pad_token: int = 0,
        dec_dropout: float = 0.1,
        dec_num_heads: int = 8,
        dec_num_blocks: int = 6,
        dec_d_ff: int = 2048,
        dec_use_bias: bool = True,
        dec_pad_token: int = 0,
    ):
        """Initializer for encoder class.

        Args:
            enc_vocab_size (int): Dimension of vocabulary for encoder for source
                sequence as per the given tokenizer and dataset.
            dec_vocab_size (int): Dimension of vocabulary for decoder for target
                sequence as per the given tokenizer and dataset.
            d_model (int, optional): Dimension for embeddings in model.
                Defaults to 512.
            max_len (int, optional): Maximum possible sequence length from
                dataset. Defaults to 3000.
            pos_dropout (float, optional): Dropout rate used in positional
                embedding layer. Defaults to 0.1.
            enc_dropout (float, optional): Dropout rate used across encoder.
            enc_num_heads (int, optional): Number of parallel heads in Multi head
                attention block in encoder. Defaults to 8.
            enc_num_blocks (int, optional): Number of encoder blocks in Decoder part
                of the Transformers. Defaults to 6.
            enc_d_ff (int, optional): Dimensionality for hidden layer of Feedforward
                layer in encoder. Defaults to 2048.
            enc_use_bias (bool, optional): Use bias for all encoder layers.
                Defaults to True.
            enc_pad_token (int, optional): Pad token to be used for masked
                softmax in encoder. Defaults to 0.
            dec_dropout (float, optional): Dropout rate used across decoder.
            dec_num_heads (int, optional): Number of parallel heads in Multi head
                attention block in decoder. Defaults to 8.
            dec_num_blocks (int, optional): Number of decoder blocks in Decoder part
                of the Transformers. Defaults to 6.
            dec_d_ff (int, optional): Dimensionality for hidden layer of Feedforward
                layer in decoder. Defaults to 2048.
            dec_use_bias (bool, optional): Use bias for all decoder layers.
                Defaults to True.
            dec_pad_token (int, optional): Pad token to be used for masked
                softmax in decoder. Defaults to 0.
        """
        super().__init__()

        self.d_model = d_model
        self.enc_vocab_size = enc_vocab_size
        self.dec_vocab_size = dec_vocab_size
        self.max_len = max_len
        self.pos_dropout = pos_dropout
        self.enc_dropout = enc_dropout
        self.enc_num_heads = enc_num_heads
        self.enc_d_k = self.enc_d_v = self.d_model // self.enc_num_heads
        self.enc_num_blocks = enc_num_blocks
        self.enc_d_ff = enc_d_ff
        self.enc_use_bias = enc_use_bias
        self.enc_pad_token = enc_pad_token

        self.dec_dropout = dec_dropout
        self.dec_num_heads = dec_num_heads
        self.dec_d_k = self.dec_d_v = self.d_model // self.dec_num_heads
        self.dec_num_blocks = dec_num_blocks
        self.dec_d_ff = dec_d_ff
        self.dec_use_bias = dec_use_bias
        self.dec_pad_token = dec_pad_token

        self.enc_emb_layer = Embeddings(self.enc_vocab_size, self.d_model)
        self.enc_pos_emb_layer = PositionalEmbeddings(
            self.d_model, self.pos_dropout, self.max_len
        )
        self.encoders = Encoder(
            self.d_model,
            self.enc_d_k,
            self.enc_d_v,
            self.enc_d_ff,
            self.enc_num_heads,
            self.enc_num_blocks,
            self.enc_use_bias,
            self.enc_dropout,
            self.enc_pad_token,
        )

        self.dec_emb_layer = Embeddings(self.dec_vocab_size, self.d_model)
        self.dec_pos_emb_layer = PositionalEmbeddings(
            self.d_model, self.pos_dropout, self.max_len
        )
        self.decoders = Decoder(
            self.d_model,
            self.dec_d_k,
            self.dec_d_v,
            self.dec_d_ff,
            self.dec_num_heads,
            self.dec_num_blocks,
            self.dec_use_bias,
            self.dec_dropout,
            self.dec_pad_token,
        )

        self.decoder_generator = DecoderGenerator(
            self.dec_vocab_size, self.d_model, apply_softmax=False
        )

        # We will reuse the decoder embedding layer weight to project the prediction
        # back to vocabulary space.
        self.decoder_generator.linear.weight = self.dec_emb_layer.emb_layer.weight

    def encoder(self, enc_seq: torch.Tensor, enc_mask: torch.Tensor) -> torch.Tensor:
        # enc_seq   : [batch, seq_len]
        # enc_mask  : [batch, 1, 1, seq_len] for encoder

        enc_repr = self.enc_emb_layer(enc_seq)  # [batch, seq_len, emb_size]
        enc_repr = self.enc_pos_emb_layer(enc_repr)  # [batch, seq_len, emb_size]
        enc_repr = self.encoders(enc_repr, enc_mask)  # [batch, seq_len, emb_size]
        return enc_repr

    def decoder(
        self,
        enc_repr: torch.Tensor,
        dec_seq: torch.Tensor,
        enc_mask: torch.Tensor,
        dec_mask: torch.Tensor,
    ) -> torch.Tensor:
        # enc_seq   : [batch, seq_len]
        # dec_seq   : [batch, seq_len]
        # enc_mask  : [batch, 1, 1, seq_len] for encoder-decoder cross attention
        # dec_mask  : [batch, 1, seq_len, seq_len] for decoder self attention

        dec_repr = self.dec_emb_layer(dec_seq)  # [batch, seq_len, emb_size]
        dec_repr = self.dec_pos_emb_layer(dec_repr)  # [batch, seq_len, emb_size]
        dec_repr = self.decoders(
            enc_repr, dec_repr, enc_mask, dec_mask
        )  # [batch, seq_len, emb_size]

        dec_logits = self.decoder_generator(dec_repr)  # [batch, seq_len, vocab_size]
        return dec_logits

    def forward(
        self,
        enc_seq: torch.Tensor,
        dec_seq: torch.Tensor,
        enc_mask: torch.Tensor,
        dec_mask: torch.Tensor,
    ) -> torch.Tensor:
        # enc_seq   : [batch, seq_len]
        # dec_seq   : [batch, seq_len]
        # enc_mask  : [batch, 1, 1, seq_len] for encoder
        # dec_mask  : [batch, 1, seq_len, seq_len] for decoder

        enc_repr = self.encoder(enc_seq, enc_mask)
        dec_logits = self.decoder(enc_repr, dec_seq, enc_mask, dec_mask)
        return dec_logits
