##
# @author Meet Patel <>
# @file Description
# @desc Created on 2023-07-05 11:38:44 pm
# @copyright MIT License
#

import os
import random
from typing import List, Tuple

import datasets
import torch
from tokenizers import Tokenizer
from tokenizers.processors import TemplateProcessing
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Sampler

from misc.utils import get_dataset_iterators
from model.utils import get_src_pad_mask, get_trg_pad_mask


def load_tokenizer(tokenizer_path: str) -> Tokenizer:
    """Function to load the tokenizer from given path and set its
    formatting.

    Args:
        tokenizer_path (str): Path of the trained tokenizer

    Returns:
        Tokenizer: Pretrained tokenizer object
    """
    if not os.path.isfile(tokenizer_path):
        print(f"No tokenizer found at location: {tokenizer_path}")

    tokenizer = Tokenizer.from_file(tokenizer_path)

    tokenizer.post_processor = TemplateProcessing(
        single="[SOS] $A [EOS]",
        special_tokens=[
            ("[SOS]", tokenizer.token_to_id("[SOS]")),
            ("[EOS]", tokenizer.token_to_id("[EOS]")),
        ],
    )

    tokenizer.enable_padding(pad_id=tokenizer.token_to_id("[PAD]"), pad_token="[PAD]")
    return tokenizer


class BatchSamplerSimilarLength(Sampler):
    def __init__(
        self,
        dataset_iterator: datasets.Dataset,
        tokenizer_de: str,
        batch_size: int,
        seq_len: int,
        shuffle: bool = True,
    ):
        """Initializer to load the dataset and sort as per the source sequences
        (german sequence) and prepare the indices in a bucketed manner where the
        sequences with similar lengths are grouped together.

        Args:
            dataset_iterator (datasets.Dataset): Dataset iterator.
            tokenizer_de (str): Source sequence tokenizer. e.g. German tokenizer.
            batch_size (int): Batch size to be used to compute upper limit of
                tokens.
            seq_len (int): Sequence length to be used to compute upper limit of
                tokens.
            shuffle (bool, optional): Shuffle the dataset before sorting and
                after getting the buckets. Defaults to True.
        """
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.total_tokens_in_batch = self.batch_size * self.seq_len
        self.shuffle = shuffle
        # get the indices and length

        # Considering only german sequences for sorting purposes.
        self.indices = [
            (i, len(tokenizer_de.encode(data["translation"]["de"]).ids))
            for i, data in enumerate(dataset_iterator)
        ]

        if self.shuffle:
            random.shuffle(self.indices)

        sorted_indices = sorted(self.indices, key=lambda x: x[1])

        self.all_batch_idx = []
        single_batch_idx = []
        cummulative_token_len = 0

        for idx, token_len in sorted_indices:
            cummulative_token_len += token_len

            single_batch_idx.append(idx)

            if cummulative_token_len > self.total_tokens_in_batch:
                self.all_batch_idx.append(single_batch_idx.copy())
                single_batch_idx.clear()
                cummulative_token_len = 0

        if self.shuffle:
            random.shuffle(self.all_batch_idx)

    def __iter__(self) -> List[int]:
        """Function will fetch list of indices to be used to generate a batch.

        Yields:
            List[int]: Yields list of indices for batch generation.
        """
        for batch_idx in self.all_batch_idx:
            random.shuffle(batch_idx)
            yield batch_idx

    def __len__(self) -> int:
        """Function to get the total number of batches which can be generated.

        Returns:
            int: Number of batches from the given dataset.
        """
        return len(self.all_batch_idx)


class DataloaderHelper:
    def __init__(
        self,
        dataset_type: str,
        tokenizer_path_en: str,
        tokenizer_path_de: str,
        batch_size: int = 8,
        seq_len: int = 128,
        split: str = "train",
        num_workers: int = 2,
        persistent_workers: bool = True,
    ):
        """Initializer to load the dataset, prepare for training and convert it
        into dataloader format.

        Args:
            dataset_type (str): Type of dataset to be used. Available options
                are "iwslt2017" and "wmt14"
            tokenizer_path_en (str): Path for english tokenizer.
            tokenizer_path_de (str): Path for German tokenizer.
            batch_size (int, optional): Batch size to be used. Defaults to 8.
            seq_len (int, optional): Sequence length to be used for combining
                sequences together. It will not produce all the samples of the
                batch as per this sequence length. But this value and batch size
                will provide some upper value which we can manage to load on GPU
                if all the tokens are non-pad tokens. Defaults to 128.
            split (str, optional): Type of dataset split. Defaults to "train".
            num_workers (int, optional): Number of parallel workers to be used
                for data loader. It should be twice of num GPUS. Defaults to 2.
            persistent_workers (bool, optional): Whether to reuse the same
                workers for all the iterations of the dataloader. Defaults to True.
        """
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.dataset_iterator = get_dataset_iterators(dataset_type, split)
        self.tokenizer_en = load_tokenizer(tokenizer_path_en)
        self.tokenizer_de = load_tokenizer(tokenizer_path_de)

        self.num_workers = num_workers
        self.persistent_workers = persistent_workers

    def get_iterator(self) -> DataLoader:
        dataloader = DataLoader(
            self.dataset_iterator,
            batch_sampler=BatchSamplerSimilarLength(
                dataset_iterator=self.dataset_iterator,
                tokenizer_de=self.tokenizer_de,
                batch_size=self.batch_size,
                seq_len=self.seq_len,
                shuffle=True,
            ),
            collate_fn=self.collate_batch,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
        )
        return dataloader

    def collate_batch(self, batch_data: List) -> Tuple[torch.Tensor]:
        """Function to tokenize sequences and prepare the mask for training.

        Args:
            batch_data (List): List of Tuple of source and target sequences.

        Returns:
            Tuple[torch.Tensor]: Tuple of source token, target token, source
                mask, target mask and target labels.
        """
        # src is german and tar is english.
        src_tokens, tar_tokens, tar_labels = [], [], []

        for data in batch_data:
            src_text = data["translation"]["de"]
            tar_text = data["translation"]["en"]
            src_tokenized_text = torch.tensor(self.tokenizer_de.encode(src_text).ids)
            tar_tokenized_text = torch.tensor(self.tokenizer_en.encode(tar_text).ids)

            src_tokens.append(src_tokenized_text)

            tar_tokenized_label = tar_tokenized_text[1:]
            tar_tokenized_text = tar_tokenized_text[:-1]

            tar_tokens.append(tar_tokenized_text)
            tar_labels.append(tar_tokenized_label)

        src_tokens = pad_sequence(src_tokens, batch_first=True)
        tar_tokens = pad_sequence(tar_tokens, batch_first=True)
        tar_labels = pad_sequence(tar_labels, batch_first=True)

        src_mask = get_src_pad_mask(src_tokens, self.tokenizer_de.token_to_id("[PAD]"))
        tar_mask = get_trg_pad_mask(tar_tokens, self.tokenizer_en.token_to_id("[PAD]"))
        return src_tokens, tar_tokens, src_mask, tar_mask, tar_labels


if __name__ == "__main__":
    tokenizer_path_en = "./tokenizer_data/bpe_tokenizer_en.json"
    tokenizer_path_de = "./tokenizer_data/bpe_tokenizer_de.json"
    bs = 8
    max_len = 128
    train_dataloader = DataloaderHelper(
        "iwslt2017", tokenizer_path_en, tokenizer_path_de, bs, max_len, "validation"
    )
    train_iter = train_dataloader.get_iterator()
    print("Dataloaded")
    op = next(iter(train_iter))
    print("Number of outputs: ", len(op))
    print("Shape of each outputs: ")
    print([s.shape for s in op])

    total_eps = 10
    for eps in range(total_eps):
        for i, data in enumerate(train_iter):
            print(
                f"{eps}/{total_eps} -- {i}/{len(train_iter)} -- {len(data)} -- {data[0].shape}"
            )
