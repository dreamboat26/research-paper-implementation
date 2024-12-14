from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


class SummaryHelper:
    def __init__(self, summary_path: str):
        """Initializer to create internal summary writer object.

        Args:
            summary_path (str): Summary files save path.
        """
        self.summary_writer = SummaryWriter(summary_path)
        self.is_closed = False

    def add_graph(self, model: nn.Module, ip_tensor: List[torch.Tensor]) -> None:
        """Function to add nn.Module of given model in tensorboard summary.

        Args:
            model (nn.Module): Pytorch Model as nn.Module.
            ip_tensor (List[torch.Tensor]): List of dummy input tensors to trace
                graph.
        """
        if self.is_closed:
            return
        self.summary_writer.add_graph(model, ip_tensor)

    def add_summary(self, summary_dict: Dict[str, Any], g_step: int) -> None:
        """Function to add values in tensorboard summary as per given dict.

        Args:
            summary_dict (Dict[str, Any]): Mapping of summary name and its value to store.
            g_step (int): Iteration number for given data.
        """
        if self.is_closed:
            return
        for key, value in summary_dict.items():
            if isinstance(value, float) or isinstance(value, int):
                # For scalar values,
                self.summary_writer.add_scalar(key, value, g_step)
            elif isinstance(value, np.ndarray):
                # For images
                self.summary_writer.add_image(key, value, g_step, dataformats="CHW")
            elif isinstance(value, torch.Tensor):
                # For Torch tensors
                if len(value.shape) == 0:
                    self.summary_writer.add_scalar(
                        key, value.detach().cpu().numpy(), g_step
                    )
                else:
                    raise NotImplementedError(
                        "Summary for tensors of more than 1 dims are not supported."
                    )
            else:
                print("Summary Input not identified", type(value))

            self.summary_writer.flush()

    def close(self) -> None:
        """Function to terminate the summary object and stop storing further summaries."""
        self.is_closed = True
        self.summary_writer.close()
