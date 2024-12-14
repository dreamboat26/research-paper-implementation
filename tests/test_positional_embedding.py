import torch

import config_wmt as c
from model.positional_embeddings import PositionalEmbeddings


class TestPositionalEmbeddings:
    def setup_class(self):
        self.emb_size = c.d_model
        self.batch_size = 2
        self.seq_len = 32
        self.pos_emb_layer = PositionalEmbeddings(self.emb_size, 0.1)

    def test_shapes(self):
        """Test case to check output shapes."""
        data = torch.randn(self.batch_size, self.seq_len, self.emb_size)
        res = self.pos_emb_layer(data)
        assert res.shape == data.shape

    def test_num_params(self):
        """Test case to check number of learnable parameters in the layer."""
        model_params = 0
        for name, weight in self.pos_emb_layer.named_parameters():
            model_params += weight.numel()
        assert model_params == 0, (
            "Positional embeddings should not have any " "learnable parameters."
        )
