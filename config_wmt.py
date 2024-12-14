from dataclasses import dataclass

d_model = 512
pos_dropout = 0.1  # Positional Embedding Dropout
max_len = 3000
tokenizer_path_en = "./tokenizer_data/bpe_tokenizer_en.json"
tokenizer_path_de = "./tokenizer_data/bpe_tokenizer_de.json"


@dataclass
class EncoderConfig:
    dropout: float = 0.1
    num_heads: int = 8
    num_block: int = 6
    d_ff: int = 2048
    use_bias: bool = True
    vocab_size: int = 30000


@dataclass
class DecoderConfig:
    dropout: float = 0.1
    num_heads: int = 8
    num_block: int = 6
    d_ff: int = 2048
    use_bias: bool = True
    vocab_size: int = 40000


@dataclass
class TrainingConfig:
    batch_size: int = 8
    seq_len: int = 128
    use_amp: bool = False
    epochs: int = 10
    train_data_path: str = "./training_data/"
    loss_logging_frequency: int = 10
    lr_scheduler: str = "cosine"
    label_smoothing: float = 0.1
    init_lr: float = 5e-4
    warmup_epochs: int = 2
    dataset: str = "wmt14"
    num_workers: int = 4
    persistent_workers: bool = True
    apply_grad_clipping: bool = False
    grad_clipping_max_norm: float = 2
    track_gradients: bool = True
    use_grad_accumulation: bool = True
    grad_accumulation_steps: int = 32  # This will make the effective batch size = batch_size * grad_accumulation_steps
    use_tpu: bool = False


enc = EncoderConfig()
dec = DecoderConfig()
train_cfg = TrainingConfig()
