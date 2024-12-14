##
# @author Meet Patel <>
# @file Description
# @desc Created on 2023-03-24 11:29:23 pm
# @copyright MIT License
#

import torch

import config_wmt as c
from model.decoder import Decoder, DecoderGenerator


class TestDecoder:
    def setup_class(self):
        self.batch_size = 2
        self.seq_len = 128
        self.d_model = c.d_model
        self.emb_size = self.d_model
        self.num_heads = c.dec.num_heads
        self.dec_vocab_size = c.dec.vocab_size
        self.d_k = self.d_v = self.d_model // self.num_heads
        self.n_blocks = c.dec.num_block
        self.d_ff = c.dec.d_ff
        self.use_bias = c.dec.use_bias
        self.dropout = c.dec.dropout
        self.decoder = Decoder(
            self.d_model,
            self.d_k,
            self.d_v,
            self.d_ff,
            self.num_heads,
            self.n_blocks,
            self.use_bias,
            self.dropout,
        )
        self.decoder_generator = DecoderGenerator(
            self.dec_vocab_size, self.d_model, False
        )

    def test_shapes(self):
        """Test case to check output shapes."""
        enc_data = torch.randn(self.batch_size, self.seq_len, self.emb_size)
        dec_data = torch.randn(self.batch_size, self.seq_len, self.emb_size)
        enc_mask = torch.ones(size=[self.batch_size, 1, 1, self.seq_len])
        enc_mask[:, :, :, (self.seq_len // 2) :] = 0
        dec_mask = torch.tril(
            torch.ones(size=[self.batch_size, 1, self.seq_len, self.seq_len])
        )

        res = self.decoder(enc_data, dec_data, enc_mask, dec_mask)
        assert list(res.shape) == [self.batch_size, self.seq_len, self.d_model]

        logits = self.decoder_generator(res)
        assert list(logits.shape) == [
            self.batch_size,
            self.seq_len,
            self.dec_vocab_size,
        ]

    def test_num_params(self):
        """Test case to check number of learnable parameters in the layer."""
        for name, weight in self.decoder.named_parameters():
            model_params = weight.numel()
            if "weight" in name:
                if "linear_1" in name or "linear_2" in name:
                    assert model_params == self.emb_size * self.d_ff
                elif "layer_norm" in name:
                    assert model_params == self.emb_size
                else:
                    assert model_params == self.emb_size * self.emb_size
            elif "bias" in name:
                if "linear_1" in name:
                    assert model_params == self.d_ff
                else:
                    assert model_params == self.emb_size
            else:
                assert False, (
                    f"Weight {name} is extra added in the model "
                    "parameters. Please check."
                )

        for name, weight in self.decoder_generator.named_parameters():
            model_params = weight.numel()
            if "weight" in name:
                assert model_params == self.emb_size * self.dec_vocab_size
            elif "bias" in name:
                assert model_params == self.dec_vocab_size
