
# Call this file as : python -m misc.plot_pos_emb --op_path "./plot_pos_emb.jpg"

import argparse

import matplotlib.pyplot as plt

from model.positional_embeddings import PositionalEmbeddings

parser = argparse.ArgumentParser(
    description="Argument parser for plotting positional embeddings."
)
parser.add_argument(
    "--op_path",
    type=str,
    required=True,
    help="Output path of positional embedding image.",
)
parser.add_argument(
    "--seq_len",
    type=int,
    default=256,
    help="Sequence length for positional embeddings.",
)
parser.add_argument(
    "--emb_dim",
    type=int,
    default=512,
    help="Embedding dimention for positional embeddings.",
)

args = parser.parse_args()
pos_emb_layer = PositionalEmbeddings(args.emb_dim, 0.1, args.seq_len)
pos_emb = pos_emb_layer.pe.detach().cpu().numpy()[0]

plt.imshow(pos_emb, cmap="hot")
plt.colorbar()
plt.savefig(args.op_path)
