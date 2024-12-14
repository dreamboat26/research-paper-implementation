import torch

import config_wmt as c
from model.attention import MultiHeadAttention


class TestMultiHeadAttention:
    def setup_class(self):
        self.batch_size = 2
        self.seq_len = 128
        self.emb_size = c.d_model
        self.d_k = self.d_v = self.emb_size // c.enc.num_heads
        self.drop_prob = c.enc.dropout
        self.num_heads = c.enc.num_heads
        self.pad_token = 0
        self.atten_block = MultiHeadAttention(
            self.emb_size,
            self.d_k,
            self.d_v,
            self.drop_prob,
            self.num_heads,
            bias=True,
            pad_token=self.pad_token,
        )

    def test_shapes(self):
        """Test case to check output shapes."""

        # masked multi head self attention
        mask = torch.ones(size=[self.batch_size, 1, 1, self.seq_len])
        x = torch.randn(self.batch_size, self.seq_len, self.emb_size)
        res = self.atten_block(x, x, x, mask)
        assert x.shape == res.shape

        # masked multi head self attention
        mask_1 = torch.ones(size=[self.batch_size, 1, 1, self.seq_len]).to(torch.int64)
        mask_1[:, :, :, (self.seq_len // 2) :] = 0
        mask_2 = torch.tril(
            torch.ones(size=[self.batch_size, 1, self.seq_len, self.seq_len])
        ).to(torch.int64)
        mask = mask_1 & mask_2
        y = torch.randn(self.batch_size, self.seq_len, self.emb_size)
        res = self.atten_block(y, y, y, mask=mask)
        assert y.shape == res.shape

        # multi head cross attention
        y = torch.randn(self.batch_size, self.seq_len, self.emb_size)
        mask = torch.ones(size=[self.batch_size, 1, 1, self.seq_len])
        mask[:, :, :, (self.seq_len // 2) :] = 0
        res = self.atten_block(y, x, x, mask)
        assert y.shape == res.shape

    def test_num_params(self):
        """Test case to check number of learnable parameters in the layer."""
        for name, weight in self.atten_block.named_parameters():
            model_params = weight.numel()
            if "weight" in name:
                assert model_params == self.emb_size * self.emb_size
            elif "bias" in name:
                assert model_params == self.emb_size
            else:
                assert False, (
                    f"Weight {name} is extra added in the model "
                    "parameters. Please check."
                )
