
from dataclasses import dataclass

d_model = 192
pos_dropout = 0.1  # Positional Embedding Dropout
max_len = 3000
tokenizer_path_en = "./tokenizer_data/bpe_tokenizer_en.json"
tokenizer_path_de = "./tokenizer_data/bpe_tokenizer_de.json"


@dataclass
class EncoderConfig:
    dropout: float = 0.1
    num_heads: int = 6
    num_block: int = 2
    d_ff: int = 256
    use_bias: bool = True
    vocab_size: int = 30000


@dataclass
class DecoderConfig:
    dropout: float = 0.1
    num_heads: int = 6
    num_block: int = 2
    d_ff: int = 256
    use_bias: bool = True
    vocab_size: int = 40000


@dataclass
class TrainingConfig:
    batch_size: int = 6
    seq_len: int = 128
    use_amp: bool = True
    epochs: int = 50
    train_data_path: str = "./training_data/"
    loss_logging_frequency: int = 100
    lr_scheduler: str = "cosine"
    label_smoothing: float = 0.1
    init_lr: float = 1e-3
    burn_in_epochs: int = 2
    dataset: str = "iwslt2017"
    num_workers: int = 2
    persistent_workers: bool = True
    apply_grad_clipping: bool = True
    grad_clipping_max_norm: float = 2
    track_gradients: bool = True
    use_grad_accumulation: bool = True
    grad_accumulation_steps: int = 8  # This will make the effective batch size = batch_size * grad_accumulation_steps


enc = EncoderConfig()
dec = DecoderConfig()
train_cfg = TrainingConfig()
