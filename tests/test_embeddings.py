##
# @author Meet Patel <>
# @file Description
# @desc Created on 2023-03-28 11:52:19 pm
# @copyright MIT License
#

import torch

import config_wmt as c
from model.embeddings import Embeddings


class TestEmbeddings:
    def setup_class(self):
        self.batch_size = 4
        self.seq_len = 128
        self.vocab_size = c.enc.vocab_size
        self.emb_dims = c.d_model
        self.emb_layer = Embeddings(self.vocab_size, self.emb_dims)

    def test_shapes(self):
        """Test case to check output shapes."""
        x = torch.randint(
            low=0,
            high=self.vocab_size,
            size=[self.batch_size, self.seq_len],
            dtype=torch.int32,
        )
        res = self.emb_layer(x)
        assert list(res.shape) == [self.batch_size, self.seq_len, self.emb_dims]

    def test_num_params(self):
        """Test case to check number of learnable parameters in the layer."""
        model_params = 0
        for name, weight in self.emb_layer.named_parameters():
            model_params += weight.numel()
        assert (
            model_params == self.vocab_size * self.emb_dims
        ), "Embeddings layer parameters are not matching."
