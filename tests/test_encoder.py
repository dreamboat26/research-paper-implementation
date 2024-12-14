import torch

import config_wmt as c
from model.encoder import Encoder


class TestEncoder:
    def setup_class(self):
        self.batch_size = 2
        self.seq_len = 128
        self.d_model = c.d_model
        self.emb_size = self.d_model
        self.num_heads = c.enc.num_heads
        self.d_k = self.d_v = self.d_model // self.num_heads
        self.n_blocks = c.enc.num_block
        self.d_ff = c.enc.d_ff
        self.use_bias = c.enc.use_bias
        self.dropout = c.enc.dropout
        self.encoder = Encoder(
            self.d_model,
            self.d_k,
            self.d_v,
            self.d_ff,
            self.num_heads,
            self.n_blocks,
            self.use_bias,
            self.dropout,
            pad_token=0,
        )

    def test_shapes(self):
        """Test case to check output shapes."""
        data = torch.randn(self.batch_size, self.seq_len, self.emb_size)
        mask = torch.ones(size=[self.batch_size, 1, 1, self.seq_len])
        mask[:, :, :, (self.seq_len // 2) :] = 0
        res = self.encoder(data, mask)
        assert res.shape == data.shape

    def test_num_params(self):
        """Test case to check number of learnable parameters in the layer."""
        for name, weight in self.encoder.named_parameters():
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
