##
# @author Meet Patel <>
# @file Description
# @desc Created on 2023-07-08 5:54:16 pm
# @copyright MIT License
#

import datetime
import importlib
import logging
import os
from dataclasses import asdict
from typing import Dict, List, Union

import wandb
import datasets
import torch
import numpy as np
from datasets import load_dataset
from torchmetrics.text import BLEUScore

from misc.logger import Logger
from misc.lr_utils.cosine_annealing_lr import CosineAnnealing
from misc.lr_utils.exp_decay_lr import ExpDecay
from misc.lr_utils.lr_scheduler import LearningRateScheduler


def get_exp_path(train_data_dir: str) -> str:
    """Function to get the directory to same the experiment related data.

    Args:
        train_data_dir (str): Directory to store all experiments.

    Returns:
        str: Path for current experiment.
    """
    start_time = datetime.datetime.now()
    exp_name = start_time.strftime("%Y_%m_%d_%H_%M_%S")
    cur_exp_path = os.path.join(train_data_dir, exp_name)
    os.makedirs(cur_exp_path, exist_ok=True)
    return cur_exp_path


def get_logger(cur_exp_path: str) -> logging.Logger:
    """Function to initialize the logger instance for current experiment.

    Args:
        cur_exp_path (str): Directory where the logger's data to be saved.

    Returns:
        logging.Logger: Logger instance.
    """
    logger_path = os.path.join(cur_exp_path)
    logger = Logger(logger_path)

    start_time = datetime.datetime.now()
    logger.info(
        "Experiment Start time: {}".format(start_time.strftime("%Y-%m-%d %H:%M:%S"))
    )
    logger.debug("Experiment files are saved at: {}".format(cur_exp_path))
    logger.debug("Training initialization completed.")
    return logger


def get_ckpt_dir(exp_path: str) -> str:
    """Function to get the path where the model is to be saved.

    Args:
        exp_path (str): Path of current experiment directory.

    Returns:
        str: Checkpoint directory path to be used.
    """
    ckpt_dir = os.path.join(exp_path, "models")
    os.makedirs(ckpt_dir, exist_ok=True)
    return ckpt_dir


def get_summary_path(exp_path: str, type: str = "train") -> str:
    """Function to get the summary path.

    Args:
        exp_path (str): Path of current experiment directory.

    Returns:
        str: Summary directory path.
    """
    summ_path = os.path.join(exp_path, type)
    os.makedirs(summ_path, exist_ok=True)
    return summ_path


def get_lr_scheduler(scheduler_type: str, *args, **kwargs) -> LearningRateScheduler:
    """Function to get learning rate scheduler based on provided type.

    Args:
        scheduler_type (str): Type of scheduler.

    Raises:
        NotImplementedError: If the type of scheduler is not implemented then this
            will be raised.

    Returns:
        LearningRateScheduler: Scheduler instance.
    """
    mapping = {"cosine": CosineAnnealing, "exp": ExpDecay}

    scheduler_class = mapping.get(scheduler_type, None)
    if scheduler_class is None:
        raise NotImplementedError(
            f"Scheduler of type: {scheduler_type} is not implemented."
        )

    return scheduler_class(*args, **kwargs)


def to_device(
    tensors: Union[torch.Tensor, List[torch.Tensor]], device=torch.device("cuda:0")
) -> Union[torch.Tensor, List[torch.Tensor]]:
    """Function to transfer single tensor or list of tensors to Device.

    Args:
        tensors (Union[torch.Tensor, List[torch.Tensor]]): Single torch tensor
            or list of torch tensors to be transfered.

    Returns:
        Union[torch.Tensor, List[torch.Tensor]]: Updated torch tensor references.
    """
    if not isinstance(tensors, List):
        return tensors.to(device, non_blocking=True)

    for i in range(len(tensors)):
        tensors[i] = tensors[i].to(device, non_blocking=True)
    return tensors


def load_module(module_path: str):
    """Function to load module from a given path.

    Args:
        module_path (str): Path of the module or python file to be imported.

    Returns:
        Upon successful import of module the imported object is returned,
            else None is returned.
    """
    try:
        if not os.path.isfile(module_path):
            print("Module not found.")
            raise FileNotFoundError
        module_name = os.path.splitext(os.path.basename(module_path))[0]
        dir_name = os.path.dirname(module_path)
        module = importlib.import_module(module_name, package=dir_name)
        return module
    except ImportError:
        return None


def get_dataset_iterators(
    dataset_type: str = "iwslt2017", split: str = "train"
) -> datasets.Dataset:
    """Function to get the training iterators for IWSLT2016 / WMT14 dataset.

    Args:
        dataset_type (str, optional): Type of dataset to be used. Defaults to "iwslt2017".
            Available options are "iwslt2017" and "wmt14"
        split (str, optional): Type of dataset split to be obtained. Defaults to "train".
    Returns:
        datasets.Dataset: Dataset iterator based on given split.
    """
    if dataset_type == "iwslt2017":
        dataset_iterator = load_dataset("iwslt2017", "iwslt2017-de-en", split=split)
    elif dataset_type == "wmt14":
        dataset_iterator = load_dataset("wmt14", "de-en", split=split)
    else:
        raise NotImplementedError(f"Provided dataset {dataset_type} not supported.")
    return dataset_iterator


def get_config_as_dict(config) -> Dict:
    """Convert the given config module into a Dict.

    Args:
        config (Module): Imported config module.

    Returns:
        Dict: Dict representation of config.
    """
    attribute_list = [
        attr
        for attr in dir(config)
        if not callable(getattr(config, attr)) and not attr.startswith("__")
    ]

    config_as_dict = {}
    for attribute in attribute_list:
        attribute_value = getattr(config, attribute)
        config_as_dict[attribute] = attribute_value
        if hasattr(attribute_value, "__dataclass_fields__"):
            config_as_dict[attribute] = asdict(attribute_value)

    return config_as_dict


def init_wandb(config, resume_wandb_id: int) -> None:
    """Initiate the weights and bias tracking. To be called at the start of experiment.

    Args:
        config (Module): Imported config module.
        resume_wandb_id (int): Weights and Bias tracking id to be reused in 
            case of resuming training. Defaults to None.
    """
    config_dict = get_config_as_dict(config)
    wandb.init(project="Attention is all you need implementation.", config=config_dict, resume="allow", id=resume_wandb_id)


def compute_bleu(prediction: Union[List[str], str], label: Union[List[str], str]) -> float:
    """Compute bleu score between prediction and label pair. Make sure that the 
    prediction and label are free from special tokens such as [SOS], [EOS], [PAD].

    Args:
        prediction (Union[List[str], str]): Prediction string. 
        label (Union[List[str], str]): Label string.

    Returns:
        float: BLEU score w.r.t. given prediction and label pair.
    """
    # We would compute BLEU for unigram, bigram, 3-gram and 4-gram.
    bleu = BLEUScore(n_gram=4)
    if isinstance(prediction, List) and isinstance(label, List):
        total_score = []
        for _pred, _label in zip(prediction, label):
            score = bleu([_pred], [[_label]])
            total_score.append(score)
        return sum(total_score) / len(total_score)
    else:
        score = bleu([prediction], [[label]])
        return score

def decode_tokens(tokenizer, batch_tokens):
    if len(batch_tokens.shape) == 2:
        # tokens : [batch, seq_len]        
        decoded_sentences = []
        for single_sentence_tokens in batch_tokens:
            single_sentence = tokenizer.decode(single_sentence_tokens)
            decoded_sentences.append(single_sentence)
        return decoded_sentences
    
    single_sentence = tokenizer.decode(batch_tokens)
    return single_sentence


class LossAverager:
    def __init__(self, num_elements: int):
        """LossAverager class initializer. It initializes internal variables for
        easy computation of average at n-th step.

        Args:
            num_elements (int): Number of elements to track simultaneously.
        """
        self.num_elements = num_elements
        self.reset()

    def reset(self):
        """Function to reset the internal variables."""
        self.val = [0 for _ in range(self.num_elements)]
        self.count = [0 for _ in range(self.num_elements)]
        self.sum = [0 for _ in range(self.num_elements)]
        self.avg = [0 for _ in range(self.num_elements)]

    def __call__(self, val_list: List[int]) -> None:
        """Function to update the internal variables as per n-th step values.
        It will also update the running mean for all the variables.

        Args:
            val_list (List[int]): List of values for each element to track.
        """
        assert len(val_list) == self.num_elements

        for i, val in enumerate(val_list):
            self.val[i] = val
            self.sum[i] += self.val[i]
            self.count[i] += 1
            self.avg[i] = self.sum[i] / self.count[i]

    def get_averages(self) -> List[float]:
        """Function to get the running average values for all the elements.

        Returns:
            List[float]: List of average values for each element.
        """
        return self.avg
