import torch

import config_wmt as c
from model.pointwise_feedforward import PointwiseFeedForwardNetwork


class TestPointwiseFeedforward:
    def setup_class(self):
        self.batch_size = 2
        self.seq_len = 128
        self.emb_size = c.d_model
        self.d_ff = c.enc.d_ff
        self.point_ffnet = PointwiseFeedForwardNetwork(self.emb_size, self.d_ff)

    def test_shapes(self):
        """Test case to check output shapes."""
        x = torch.randn(self.batch_size, self.seq_len, self.emb_size)
        res = self.point_ffnet(x)
        assert x.shape == res.shape

    def test_num_params(self):
        """Test case to check number of learnable parameters in the layer."""
        for name, weight in self.point_ffnet.named_parameters():
            model_params = weight.numel()
            if "weight" in name:
                assert model_params == self.emb_size * self.d_ff
            elif "bias" in name and "linear_1" in name:
                assert model_params == self.d_ff
            elif "bias" in name and "linear_2" in name:
                assert model_params == self.emb_size
            else:
                assert False, (
                    f"Weight {name} is extra added in the model "
                    "parameters. Please check."
                )
