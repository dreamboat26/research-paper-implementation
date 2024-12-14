import argparse
from typing import Tuple

import torch
import torch.nn.functional as F
from torch.cuda.amp.grad_scaler import GradScaler
from torch.nn import CrossEntropyLoss
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchinfo import summary
from tqdm.auto import tqdm

import wandb
from dataset_loader import DataloaderHelper, load_tokenizer
from misc.checkpoint_handler import CheckpointHandler
from misc.summary_writer import SummaryHelper
from misc.utils import (
    LossAverager,
    compute_bleu,
    decode_tokens,
    get_ckpt_dir,
    get_exp_path,
    get_logger,
    get_lr_scheduler,
    get_summary_path,
    init_wandb,
    load_module,
    to_device,
)
from model.enc_dec_transformer import TransformerEncoderDecoder


class Trainer:
    def __init__(
        self,
        cfg,
        resume: bool = False,
        resume_ckpt: str = None,
        resume_wandb_id: int = None,
    ):
        """Initializer for trainer class.

        Args:
            cfg (Module): Config Module.
            resume (bool, optional): Whether to resume the training. Defaults
                to False.
            resume_ckpt (str, optional): Checkpoint path to be used to resume
                training with. Defaults to None.
            resume_wandb_id (str, optinal): Weights and Bias tracking id to be
                reused in case of resuming training. Defaults to None.
        """
        self.cfg = cfg
        self.resume = resume
        self.resume_ckpt = resume_ckpt
        self.cpu = torch.device("cpu:0")
        self.cuda = torch.device("cuda:0")

        self.tokenizer_en = load_tokenizer(cfg.tokenizer_path_en)
        self.tokenizer_de = load_tokenizer(cfg.tokenizer_path_de)

        self.train_dataloader = DataloaderHelper(
            cfg.train_cfg.dataset,
            cfg.tokenizer_path_en,
            cfg.tokenizer_path_de,
            cfg.train_cfg.batch_size,
            cfg.train_cfg.seq_len,
            "train",
            cfg.train_cfg.num_workers,
            cfg.train_cfg.persistent_workers,
        )
        self.valid_dataloader = DataloaderHelper(
            cfg.train_cfg.dataset,
            cfg.tokenizer_path_en,
            cfg.tokenizer_path_de,
            cfg.train_cfg.batch_size,
            cfg.train_cfg.seq_len,
            "validation",
            cfg.train_cfg.num_workers,
            cfg.train_cfg.persistent_workers,
        )
        self.test_dataloader = DataloaderHelper(
            cfg.train_cfg.dataset,
            cfg.tokenizer_path_en,
            cfg.tokenizer_path_de,
            cfg.train_cfg.batch_size,
            cfg.train_cfg.seq_len,
            "test",
            cfg.train_cfg.num_workers,
            cfg.train_cfg.persistent_workers,
        )
        self.enc_pad_token = self.tokenizer_de.token_to_id("[PAD]")
        self.dec_pad_token = self.tokenizer_en.token_to_id("[PAD]")

        self.transformer = TransformerEncoderDecoder(
            cfg.enc.vocab_size,
            cfg.dec.vocab_size,
            cfg.d_model,
            cfg.max_len,
            cfg.pos_dropout,
            cfg.enc.dropout,
            cfg.enc.num_heads,
            cfg.enc.num_block,
            cfg.enc.d_ff,
            cfg.enc.use_bias,
            self.enc_pad_token,
            cfg.dec.dropout,
            cfg.dec.num_heads,
            cfg.dec.num_block,
            cfg.dec.d_ff,
            cfg.dec.use_bias,
            self.dec_pad_token,
        )

        self.epochs = self.cfg.train_cfg.epochs

        self.lr_scheduler = get_lr_scheduler(
            self.cfg.train_cfg.lr_scheduler,
            init_lr=self.cfg.train_cfg.init_lr,
            epochs=self.epochs,
            warmup_epochs=self.cfg.train_cfg.warmup_epochs,
            steps_per_epoch=len(self.train_dataloader.get_iterator()),
        )
        self.exp_path = get_exp_path(cfg.train_cfg.train_data_path)
        self.logger = get_logger(self.exp_path)
        self.ckpt_dir = get_ckpt_dir(self.exp_path)
        self.ckpt_handler = CheckpointHandler(self.ckpt_dir, "model", max_to_keep=3)
        self.accumulation_steps = (
            self.cfg.train_cfg.grad_accumulation_steps
            if self.cfg.train_cfg.use_grad_accumulation
            else 1
        )
        init_wandb(self.cfg, resume_wandb_id)

    def __print_model_summary(self) -> None:
        """Function to print the model's summary.
        e.g. Each layer's shapes and parameters.
        """
        # src_tokens: [batch, se(_len]
        # dst_tokens: [batch, seq_len]
        # src_mask  : [batch, 1, 1, seq_len]
        # dst_mask  : [batch, 1, seq_len, seq_len]
        valid_iter = self.valid_dataloader.get_iterator()
        src_tokens, dst_tokens, src_mask, dst_mask, dst_labels = next(iter(valid_iter))
        model_stats = summary(
            self.transformer,
            input_data=[src_tokens, dst_tokens, src_mask, dst_mask],
            verbose=0,
        )
        for line in str(model_stats).split("\n"):
            self.logger.debug(line)

    def __compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        loss_fn: CrossEntropyLoss,
    ) -> torch.Tensor:
        """Function to compute loss value based on the logits and labels.

        Args:
            logits (torch.Tensor): Logits from model in shape [b x seq x vocab].
            labels (torch.Tensor): Labels from dataloader in shape [b x seq].
            loss_fn (CrossEntropyLoss): Cross entropy loss function instance.

        Returns:
            torch.Tensor: Calculated loss value.
        """
        # logits: [B, seq, vocab]
        # labels: [B, seq]

        # Skipping first token in the decoder sequence as the first token is for
        # [SOS]. So we dont need to compute the loss for [SOS] input.
        labels = labels[:, 1:]
        logits = logits[:, 1:, :]

        batch_size = logits.shape[0]
        vocab_size = logits.shape[-1]

        logits = torch.reshape(logits, (-1, vocab_size))
        labels = torch.reshape(labels, (-1,))

        # We would take mean across all sequence length and all batches.
        loss = loss_fn(logits, labels) * batch_size / self.accumulation_steps

        return loss

    def __compute_accuracy_and_bleu(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor]:
        """Function to compute accuracy value and bleu value based on the
        logits and labels.

        Args:
            logits (torch.Tensor): Logits from model in shape [b x seq x vocab].
            labels (torch.Tensor): Labels from dataloader in shape [b x seq].

        Returns:
            Tuple[torch.Tensor]: Tuple of calculated accuracy and bleu value.
        """
        # logits: [B, seq, vocab]
        # labels: [B, seq]

        # Skipping first token in the decoder sequence as the first token is for
        # [SOS]. So we dont need to compute the loss for [SOS] input.

        decoded_labels = decode_tokens(
            self.tokenizer_en, to_device(labels, self.cpu).numpy()
        )
        # [N]
        decoded_logits = decode_tokens(
            self.tokenizer_en, to_device(torch.argmax(logits, dim=-1), self.cpu).numpy()
        )
        # [N]

        # all_tokens = torch.concat([labels, torch.argmax(logits, dim=-1)], dim=0)

        # decoded_tokens = decode_tokens(self.tokenizer_en, to_device(all_tokens, self.cpu).numpy())

        # decoded_labels = decoded_tokens[:len(decoded_tokens)//2]
        # # [N]
        # decoded_logits = decoded_tokens[len(decoded_tokens)//2:]
        # # [N]

        bleu = compute_bleu(decoded_logits, decoded_labels)

        labels = labels[:, 1:]
        logits = logits[:, 1:, :]

        batch_size = logits.shape[0]
        seq_len = logits.shape[1]
        vocab_size = logits.shape[-1]

        logits = torch.reshape(logits, (-1, vocab_size))
        labels = torch.reshape(labels, (-1,))

        probs = F.softmax(logits.detach(), dim=-1)

        # We need to ignore pad tokens.
        label_mask = labels != self.dec_pad_token

        # We would take mean across all sequence lengths and all batches.
        acc = (
            ((torch.argmax(probs, dim=-1) == labels) * label_mask).sum()
            * 100
            / (batch_size * seq_len)
        )

        return acc, bleu

    def __train_batch_loop(
        self,
        iterator: DataLoader,
        loss_fn: CrossEntropyLoss,
        opt: Optimizer,
        scaler: GradScaler,
        summ_writer: SummaryHelper,
        eps: int,
    ) -> None:
        """Function to run the training loop for single epoch based on given
        iterator.

        Args:
            iterator (DataLoader): Training dataloader.
            loss_fn (CrossEntropyLoss): Cross entropy loss function instance.
            opt (Optimizer): Optimizer instance.
            scaler (GradScaler): Gradient scaler instance.
            summ_writer (SummaryHelper): Summary object to store data for
                tensorboard visualization.
            eps (int): Epoch number.
        """
        self.transformer.train()

        for batch_num, (
            src_tokens,
            dst_tokens,
            src_mask,
            dst_mask,
            dst_labels,
        ) in enumerate(tqdm(iterator)):
            # src_tokens: [batch, seq_len]
            # dst_tokens: [batch, seq_len]
            # src_mask  : [batch, 1, 1, seq_len]
            # dst_mask  : [batch, 1, seq_len, seq_len]
            # dst_labels: [batch, seq_len]

            # Uncomment below line to identify the issues related to nan when
            # using AMP training.
            # with torch.autograd.detect_anomaly(check_nan=True):
            with torch.cuda.amp.autocast(enabled=self.cfg.train_cfg.use_amp):
                src_tokens, dst_tokens, src_mask, dst_mask, dst_labels = to_device(
                    [src_tokens, dst_tokens, src_mask, dst_mask, dst_labels], self.cuda
                )
                logits = self.transformer(src_tokens, dst_tokens, src_mask, dst_mask)
                loss = self.__compute_loss(logits, dst_labels, loss_fn)
                loss = loss / self.accumulation_steps

            scaler.scale(loss).backward()

            # Keeping this outside since it is used in multiple if conditions.
            lr = self.lr_scheduler.step(self.g_step, opt)

            if ((self.g_step + 1) % self.accumulation_steps == 0) or (
                self.g_step + 1 == len(iterator)
            ):
                # As per doc, `unscale_` should only be called once per optimizer
                # per `step` call and only after all gradients for that optimizer's
                # assigned parameters have been accumulated.
                # Calling `unscale_` twice for a given optimizer between each `step`
                # triggers a RuntimeError.

                if self.cfg.train_cfg.apply_grad_clipping:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(
                        opt.param_groups[0]["params"],
                        max_norm=self.cfg.train_cfg.grad_clipping_max_norm,
                    )

                scaler.step(opt)
                scaler.update()
                opt.zero_grad()

            if (batch_num + 1) % self.cfg.train_cfg.loss_logging_frequency == 0:
                loss = to_device(loss, self.cpu)

                acc, bleu = self.__compute_accuracy_and_bleu(logits, dst_labels)
                acc = to_device(acc, self.cpu)

                self.logger.info(
                    f"Epoch: {eps + 1}/{self.epochs}, "
                    f"Batch No.: {batch_num + 1}/{len(iterator)}, "
                    f"Loss: {loss:.4f}, "
                    f"Accuracy: {acc:.2f}, "
                    f"BLEU: {bleu:.4f}, "
                    f"LR: {lr:.5f}"
                )

                metrics = {
                    "Epoch": (eps + 1),
                    "Loss": loss,
                    "Accuracy": acc,
                    "BLEU": bleu,
                    "LR": lr,
                }
                wandb.log(metrics, step=self.g_step)

                summ_writer.add_summary(
                    {"loss": loss, "accuracy": acc, "lr": lr, "bleu": bleu}, self.g_step
                )
            self.g_step += 1

    def __test_batch_loop(
        self,
        iterator: DataLoader,
        loss_fn: CrossEntropyLoss,
        summ_writer: SummaryHelper,
    ) -> Tuple[float]:
        """Function to run the test loop at the end of each epoch.

        Args:
            iterator (DataLoader): Test dataloader.
            loss_fn (CrossEntropyLoss): Cross entropy loss function instance.
            summ_writer (SummaryHelper): Summary object to store data for
                tensorboard visualization.

        Returns:
            Tuple[float]: Tuple of average test loss and average test accuracy.
        """
        self.transformer.eval()

        # Only test_loss, accuracy and bleu to be tracked.
        test_tracker = LossAverager(num_elements=3)

        with torch.no_grad():
            for src_tokens, dst_tokens, src_mask, dst_mask, dst_labels in tqdm(
                iterator
            ):
                # src_tokens: [batch, seq_len]
                # dst_tokens: [batch, seq_len]
                # src_mask  : [batch, 1, 1, seq_len]
                # dst_mask  : [batch, 1, seq_len, seq_len]
                # dst_labels: [batch, seq_len]

                src_tokens, dst_tokens, src_mask, dst_mask, dst_labels = to_device(
                    [src_tokens, dst_tokens, src_mask, dst_mask, dst_labels], self.cuda
                )
                logits = self.transformer(src_tokens, dst_tokens, src_mask, dst_mask)
                loss = self.__compute_loss(logits, dst_labels, loss_fn)
                acc, bleu = self.__compute_accuracy_and_bleu(logits, dst_labels)
                test_tracker([loss, acc, bleu])

            avg_test_loss, avg_test_acc, avg_test_bleu = test_tracker.get_averages()

            self.logger.info(
                f"Avg. Test Loss: {avg_test_loss:.4f}, "
                + f"Avg. Test Accuracy: {avg_test_acc:.2f}, "
                + f"Avg. Test BLEU: {avg_test_bleu:.4f}"
            )
            metrics = {
                "Avg. Test Loss": avg_test_loss,
                "Avg. Test Accuracy": avg_test_acc,
                "Avg. Test BLEU": avg_test_bleu,
            }
            wandb.log(metrics, step=self.g_step)

            summ_writer.add_summary(
                {
                    "loss": avg_test_loss,
                    "accuracy": avg_test_acc,
                    "bleu": avg_test_bleu,
                },
                self.g_step,
            )
        return avg_test_loss, avg_test_acc, avg_test_bleu

    def __get_summary_writer(self, type: str) -> SummaryHelper:
        """Function to get the SummaryHelper object which will create and upated
        summary object during training.

        Args:
            type (str): Type of summary object. It can be "train" or "test".

        Returns:
            SummaryHelper: SummaryHelper instance as per given type.
        """
        summ_path = get_summary_path(self.exp_path, type)
        return SummaryHelper(summ_path)

    def __save_checkpoint(
        self, eps: int, loss: float, opt: Optimizer, scaler: GradScaler
    ) -> None:
        """Function to save the checkpoint along with model state, optimizer
        state, scaler state and other attributes corresponding to current epoch.

        Args:
            eps (int): Epoch number.
            loss (float): Test Loss value
            opt (Optimizer): Optimizer instance.
            scaler (GradScaler): Gradient scaler instance.
        """
        checkpoint = {
            "epoch": eps,
            "global_step": self.g_step,
            "test_loss": loss,
            "model": self.transformer.state_dict(),
            "optimizer": opt.state_dict(),
            "scaler": scaler.state_dict(),
        }
        self.ckpt_handler.save(checkpoint)

    def __resume_training(
        self, resume: bool, resume_ckpt: str, opt: Optimizer, scaler: GradScaler
    ) -> Tuple[int]:
        """Function to restore the training parameters and return resume epoch
        number and global iteration number.

        Args:
            resume (bool): True for resuming the training else False.
            resume_ckpt (str): Path of resume checkpoint to be restored.
            opt (Optimizer): Optimizer instance.
            scaler (GradScaler): Gradient scaler instance.

        Returns:
            Tuple[int]: Tuple of resume epoch number and global iteration count.
        """
        if resume:
            checkpoint = torch.load(resume_ckpt)
            self.transformer.load_state_dict(checkpoint["model"])
            opt.load_state_dict(checkpoint["optimizer"])
            scaler.load_state_dict(checkpoint["scaler"])
            resume_g_step = checkpoint["global_step"]
            resume_eps = checkpoint["epoch"]
            self.logger.info(f"Resuming training from {resume_eps} epochs.")
        else:
            resume_g_step = 1
            resume_eps = 0

        g_step = max(1, resume_g_step)
        return resume_eps, g_step

    def __get_loss_function(self) -> CrossEntropyLoss:
        """Function to get the cross entropy loss function instance.

        Returns:
            CrossEntropyLoss: Instance of cross entropy loss.
        """
        return CrossEntropyLoss(
            reduction="mean",
            label_smoothing=self.cfg.train_cfg.label_smoothing,
            ignore_index=self.dec_pad_token,
        )

    def train(self) -> None:
        """Function to be used to train the model based on provided config."""
        self.__print_model_summary()

        loss_fn = self.__get_loss_function()

        opt = torch.optim.Adam(self.transformer.parameters(), lr=0.0, weight_decay=0.0)
        scaler = torch.cuda.amp.GradScaler(enabled=self.cfg.train_cfg.use_amp)

        resume_eps, self.g_step = self.__resume_training(
            self.resume, self.resume_ckpt, opt, scaler
        )

        train_writer = self.__get_summary_writer("train")
        test_writer = self.__get_summary_writer("test")

        train_iter = self.train_dataloader.get_iterator()
        valid_iter = self.valid_dataloader.get_iterator()
        test_iter = self.test_dataloader.get_iterator()

        self.transformer.to(self.cuda)

        if self.cfg.train_cfg.track_gradients:
            wandb.watch(self.transformer)

        for eps in range(resume_eps, self.epochs):
            self.logger.info(f"Epoch: {eps + 1}/{self.epochs} Started")
            self.__train_batch_loop(train_iter, loss_fn, opt, scaler, train_writer, eps)

            avg_test_loss, avg_test_acc, avg_test_bleu = self.__test_batch_loop(
                test_iter, loss_fn, test_writer
            )

            self.__save_checkpoint(eps + 1, avg_test_loss, opt, scaler)
            self.logger.info(f"Epoch: {eps + 1}/{self.epochs} completed")

        print("Training Completed.")
        train_writer.close()
        test_writer.close()
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Argument parser for training the model."
    )
    parser.add_argument(
        "--config_path",
        action="store",
        type=str,
        required=True,
        help="Path of config file to be used for training.",
    )
    parser.add_argument(
        "--resume",
        action="store",
        type=bool,
        help="Use it to resume the training.",
    )
    parser.add_argument(
        "--resume_ckpt",
        type=str,
        default=None,
        help="Checkpoint path to be used for resuming the training.",
    )
    parser.add_argument(
        "--resume_wandb_id",
        type=str,
        default=None,
        help="Weights and biases experiment id to be used for resuming.",
    )
    args = parser.parse_args()

    config = load_module(args.config_path)
    trainer = Trainer(config, args.resume, args.resume_ckpt, args.resume_wandb_id)
    trainer.train()
