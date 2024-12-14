##
# @author Meet Patel <>
# @file Description
# @desc Created on 2023-10-18 6:43:26 pm
# @copyright MIT License
#

# Call this file as : python -m misc.plot_lr --op_path "./plot_lr.jpg"

import argparse

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from misc.utils import get_lr_scheduler

parser = argparse.ArgumentParser(
    description="Argument parser for plotting learning rate."
)
parser.add_argument(
    "--op_path",
    type=str,
    required=True,
    help="Output path of learning rate graph image.",
)
parser.add_argument(
    "--scheduler_type",
    type=str,
    default="cosine",
    help="Type of scheduler for learning rate.",
)
parser.add_argument(
    "--init_lr",
    type=float,
    default=0.001,
    help="Initial value of learning rate (excluding warmup stage).",
)
parser.add_argument(
    "--num_epochs",
    type=int,
    default=10,
    help="Number of epochs for learning rate.",
)
parser.add_argument(
    "--steps_per_epoch",
    type=int,
    default=17061,  # Based on WMT dataset
    help="Steps per epoch in training model.",
)
parser.add_argument(
    "--warmup_epochs",
    type=int,
    default=2,
    help="Number of epochs for warm up stage.",
)

args = parser.parse_args()

lr_scheduler = get_lr_scheduler(
    scheduler_type=args.scheduler_type,
    init_lr=args.init_lr,
    epochs=args.num_epochs,
    warmup_epochs=args.warmup_epochs,
    steps_per_epoch=args.steps_per_epoch,
)

lr_mapping = []
g_step = 0
for e in tqdm(range(args.num_epochs)):
    for b in range(args.steps_per_epoch):
        lr = lr_scheduler.step(g_step, None)
        g_step += 1
        if (g_step % 1000) == 0:
            lr_mapping.append([g_step, lr])
            print(f"Epoch {e}, Batch num: {b}, G_Step: {g_step}, LR: {lr:.6f}")

lr_mapping = np.array(lr_mapping)  # [N, 2]
training_steps = np.int32(lr_mapping[:, 0])  # [N]
learning_rates = lr_mapping[:, 1]  # [N]

plt.plot(training_steps, learning_rates, linestyle="-")
plt.xlabel("Training Steps")
plt.ylabel("Learning Rate")
plt.title("Learning Rate Schedule")
plt.savefig(args.op_path)
