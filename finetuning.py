"""
This code is hacked together from a lot of different codebases:
https://github.com/mkshing/svdiff-pytorch
https://github.com/XavierXiao/Dreambooth-Stable-Diffusion
https://github.com/huggingface/diffusers
https://github.com/CompVis/latent-diffusion
"""

import os
import sys
sys.path.append('./')
import argparse
import itertools
import math
from pathlib import Path
import torch
import json
import torch.nn.functional as F
import torchvision
from utils import DreamBoothDataset, freeze_parameters, sample, load_unet_for_svdiff, load_text_encoder_for_svdiff, collate_fn
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils.import_utils import is_xformers_available
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from transformers import CLIPTextModel
import retrieve
import logging
from safetensors.torch import save_file


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--user_id", type=str)
    parser.add_argument("--model_id", type=str)
    parser.add_argument("--topic_id", type=str)
    parser.add_argument("--pretrained_model_name_or_path", type=str, default=None)
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--instance_data_dir",type=str,default=None)
    parser.add_argument("--class_data_dir",type=str,default=None)
    parser.add_argument("--instance_prompt",type=str,default=None)
    parser.add_argument("--class_prompt", type=str, default=None)
    parser.add_argument("--prior_loss_weight", type=float, default=1.0)
    parser.add_argument("--num_class_images", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--center_crop", default=False,type=bool)
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--sample_batch_size", type=int, default=4)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--learning_rate_1d", type=float, default=1e-6)
    parser.add_argument("--scale_lr", default=False, type=bool)
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--optim", default="adam",type=bool)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-08)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"])
    parser.add_argument("--concepts_list", type=str, default=None)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--modifier_tokens", type=list, default=None)
    parser.add_argument("--initializer_tokens", type=list, default='ktn+pll+ucd')
    parser.add_argument("--hflip",default=False,type=bool)
    parser.add_argument("--checkpointing_steps",default=1000,type=int)
    parser.add_argument("--enable_xformers_memory_efficient_attention",default=False,type=bool)
    parser.add_argument("--enable_tomesd",default=False,type=bool)
    parser.add_argument("--resume_from_checkpoint",default=False,type=bool)
    parser.add_argument("--train_text_encoder",default=False,type=bool)

    args = parser.parse_args()    

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.train_text_encoder and args.gradient_accumulation_steps > 1 and accelerator.num_processes > 1:
        raise ValueError(
            "Gradient accumulation is not supported when training the text encoder in distributed training. "
            "Please set gradient_accumulation_steps to 1. This feature will be supported in the future."
        )

    # if args.concepts_list is None:
    #     if args.class_data_dir is None:
    #         raise ValueError("You must specify a data directory for class images.")
    #     if args.class_prompt is None:
    #         raise ValueError("You must specify prompt for class images.")
    return args


def main(args):
    logging.info(f"Finetune:{args.model_id}:Starting Finetuning")
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="tensorboard",
        logging_dir=logging_dir,
    )

    if args.seed is not None:
        set_seed(args.seed)

    with open(args.concepts_list, "r") as f:
        args.concepts_list = json.load(f)

    logging.info(f"Finetune:{args.model_id}:Concept List - {args.concepts_list}")

    for i, concept in enumerate(args.concepts_list):
        class_images_dir = Path(concept['class_data_dir'])
        if not class_images_dir.exists():
            class_images_dir.mkdir(parents=True, exist_ok=True)
        if accelerator.is_main_process:
            name = '_'.join(concept['class_prompt'].split())
            if not Path(os.path.join(class_images_dir, name)).exists() or len(list(Path(os.path.join(class_images_dir, name)).iterdir())) < args.num_class_images:
                retrieve.retrieve(concept['class_prompt'], class_images_dir, args.num_class_images)
        # concept['class_prompt'] = os.path.join(class_images_dir, f'caption.txt')
        # concept['class_data_dir'] = os.path.join(class_images_dir, f'images.txt')
        args.concepts_list[i] = concept
        accelerator.wait_for_everyone()

    # Handle the repository creation
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    # try:
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision, use_fast=False)
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision)

    if args.train_text_encoder:
        if args.resume_from_checkpoint:
            text_encoder, trained_token_embeds = load_text_encoder_for_svdiff(args.pretrained_model_name_or_path, 
                                                        subfolder="text_encoder", revision=args.revision, 
                                                        spectral_shifts_ckpt=args.resume_from_checkpoint)
        else:
            text_encoder, trained_token_embeds = load_text_encoder_for_svdiff(args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision)
    else:
        text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision)

    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)
    if args.resume_from_checkpoint:
        unet = load_unet_for_svdiff(args.pretrained_model_name_or_path,
                                    subfolder="unet", revision=args.revision, low_cpu_mem_usage=True,
                                    spectral_shifts_ckpt=args.resume_from_checkpoint)
    else:
        unet = load_unet_for_svdiff(args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, low_cpu_mem_usage=True)
    unet.requires_grad_(False)
    optim_params = []
    optim_params_1d = []
    for n, p in unet.named_parameters():
        if "delta" in n:
            p.requires_grad = True
            if "norm" in n:
                optim_params_1d.append(p)
            else:
                optim_params.append(p)

    if args.train_text_encoder:
        for n, p in text_encoder.named_parameters():
            if "delta" in n:
                p.requires_grad = True
                if "norm" in n:
                    optim_params_1d.append(p)
                else:
                    optim_params.append(p)
    total_params = sum(p.numel() for p in optim_params) + sum(p.numel() for p in optim_params_1d)
    print(f"Number of Trainable Parameters: {total_params}")
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    # except Exception as e:
    #     logging.error(f"Finetune:{args.model_id}:Error while loading models.")
    #     logging.error(f"Finetune:{args.model_id}:{e}")
    #     exit()

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            from packaging import version
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logging.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")
        
    if args.enable_tomesd:
        try:
            import tomesd
        except ImportError:
            raise ImportError(
                "To use token merging (ToMe), please install the tomesd library: `pip install tomesd`."
            )
        tomesd.apply_patch(unet, ratio=0.5)

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )
        args.learning_rate = args.learning_rate*2.

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.optim == "bitsandbytes":
        import bitsandbytes as bnb
        optimizer_class = bnb.optim.AdamW8bit
    elif args.optim == "lion":
        optimizer_class = Lion
    else:
        optimizer_class = torch.optim.AdamW

    logging.info(f"Finetune:{args.model_id}:All models loaded")
    
    # Adding a modifier token which is optimized ####
    # Code taken from https://github.com/huggingface/diffusers/blob/main/examples/textual_inversion/textual_inversion.py
    modifier_token_ids = []
    initializer_token_ids = []
    if args.modifier_tokens:
        logging.info(f"Finetune:{args.model_id}:Modifier Token - {args.modifier_tokens}")
        if len(args.modifier_tokens) > len(args.initializer_tokens):
            raise ValueError("You must specify + separated initializer token for each modifier token.")
        for modifier_token, initializer_token in zip(args.modifier_tokens, args.initializer_tokens[:len(args.modifier_tokens)]):
            # Add the placeholder token in tokenizer
            num_added_tokens = tokenizer.add_tokens(modifier_token)
            if num_added_tokens == 0:
                logging.error(f"Finetune:{args.model_id}:Token '{modifier_token}' already exists in tokenizer. Use a different one.")
                raise ValueError(f"Token '{modifier_token}' already exists in tokenizer. Use a different one.")

            # Convert the initializer_token, placeholder_token to ids
            token_ids = tokenizer.encode([initializer_token], add_special_tokens=False)
            logging.info(f"Finetune:{args.model_id}:Token IDs - {token_ids}")

            # Check if initializer_token is a single token or a sequence of tokens
            if len(token_ids) > 1:
                logging.error(f"Finetune:{args.model_id}:The initializer token must be a single token.")
                raise ValueError("The initializer token must be a single token.")

            initializer_token_ids.append(token_ids[0])
            modifier_token_ids.append(tokenizer.convert_tokens_to_ids(modifier_token))

        # Resize the token embeddings as we are adding new special tokens to the tokenizer
        text_encoder.resize_token_embeddings(len(tokenizer))
        # Initialise the newly added placeholder token with the embeddings of the initializer token
        token_embeds = text_encoder.get_input_embeddings().weight.data
        for modifier_token, modifier_token_id, initializer_token_id in zip(args.modifier_tokens, modifier_token_ids, initializer_token_ids):
            token_embeds[modifier_token_id] = token_embeds[initializer_token_id]
            if trained_token_embeds:
                if modifier_token in trained_token_embeds.keys():
                    token_embeds[modifier_token_id] = trained_token_embeds[modifier_token]
                    logging.info(f"Finetune:{args.model_id}:Loaded {modifier_token_id} token embedding from previous session.")
                    print(f"Finetune:{args.model_id}:Loaded {modifier_token_id} token embedding from previous session.")

        # Freeze all parameters except for the token embeddings in text encoder
        params_to_freeze = itertools.chain(
            text_encoder.text_model.encoder.parameters(),
            text_encoder.text_model.final_layer_norm.parameters(),
            text_encoder.text_model.embeddings.position_embedding.parameters(),
        )
        freeze_parameters(params_to_freeze)

        params_to_optimize = itertools.chain(text_encoder.get_input_embeddings().parameters(), optim_params)
        logging.info(f"Finetune:{args.model_id}:Successfully added all modifier tokens.")
    else:
        logging.info(f"Finetune:{args.model_id}:Did not receive any modifier tokens, finetuning without.")
        params_to_optimize = (
            itertools.chain(optim_params) 
        )

    optimizer = optimizer_class(
        [{"params": params_to_optimize}, {"params": optim_params_1d, "lr": args.learning_rate_1d}],
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    # optimizer = optimizer_class(params_to_optimize)

    noise_scheduler = DDPMScheduler.from_config(args.pretrained_model_name_or_path, subfolder="scheduler")

    train_dataset = DreamBoothDataset(
        concepts_list=args.concepts_list,
        num_class_images=args.num_class_images,
        tokenizer=tokenizer,
        size=args.resolution,
        center_crop=args.center_crop,
    )

    logging.info(f"Finetune:{args.model_id}:Train Dataset & Noise Schedule loaded.")

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, collate_fn=lambda examples: collate_fn(examples, True), num_workers=8)

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # Check that all trainable models are in full precision
    low_precision_error_string = (
        "Please make sure to always have all model weights in full float32 precision when starting training - even if"
        " doing mixed precision training. copy of the weights should still be float32."
    )

    if accelerator.unwrap_model(unet).dtype != torch.float32:
        raise ValueError(
            f"Unet loaded as datatype {accelerator.unwrap_model(unet).dtype}. {low_precision_error_string}"
        )

    if args.train_text_encoder and accelerator.unwrap_model(text_encoder).dtype != torch.float32:
        raise ValueError(
            f"Text encoder loaded as datatype {accelerator.unwrap_model(text_encoder).dtype}."
            f" {low_precision_error_string}"
        )


    if args.modifier_tokens or args.train_text_encoder:
        unet, text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, text_encoder, optimizer, train_dataloader, lr_scheduler
        )
    else:
        unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, optimizer, train_dataloader, lr_scheduler
        )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae.to(accelerator.device, dtype=weight_dtype)
    if not args.modifier_tokens and not args.train_text_encoder:
        text_encoder.to(accelerator.device, dtype=weight_dtype)

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # if accelerator.is_main_process:
        # accelerator.init_trackers("svdiff", config=vars(args))

  # cache keys to save
    state_dict_keys = [k for k in accelerator.unwrap_model(unet).state_dict().keys() if "delta" in k]
    if args.train_text_encoder:
        state_dict_keys_te = [k for k in accelerator.unwrap_model(text_encoder).state_dict().keys() if "delta" in k]
    
    def save_weights(step, save_path=None):
        # Create the pipeline using using the trained modules and save it.
        if accelerator.is_main_process:
            if save_path is None:
                save_path = os.path.join(args.output_dir, f"checkpoint-{step}")
            os.makedirs(save_path, exist_ok=True)
            state_dict = accelerator.unwrap_model(unet, keep_fp32_wrapper=True).state_dict()
            state_dict = {k: state_dict[k] for k in state_dict_keys}
            save_file(state_dict, os.path.join(save_path, "spectral_shifts.safetensors"))
            if args.train_text_encoder:
                state_dict = accelerator.unwrap_model(text_encoder, keep_fp32_wrapper=True).state_dict()
                state_dict = {k: state_dict[k] for k in state_dict_keys_te}
                if args.modifier_tokens:
                    for i in range(len(modifier_token_ids)):
                        token_embed = accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[modifier_token_ids[i]].detach().cpu()
                        state_dict[f"modifier_tokens.{args.modifier_tokens[i]}"] = token_embed
                        print(f"Saving {args.modifier_tokens[i]} token: {token_embed.shape}.")
                # print(state_dict.keys())
                save_file(state_dict, os.path.join(save_path, "spectral_shifts_te.safetensors"))

            print(f"[*] Weights saved at {save_path}")

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logging.info("***** Running training *****")
    logging.info(f"Finetune:{args.model_id}:Num examples = {len(train_dataset)}")
    logging.info(f"Finetune:{args.model_id}:Num batches each epoch = {len(train_dataloader)}")
    logging.info(f"Finetune:{args.model_id}:Num Epochs = {args.num_train_epochs}")
    logging.info(f"Finetune:{args.model_id}:Instantaneous batch size per device = {args.train_batch_size}")
    logging.info(f"Finetune:{args.model_id}:Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logging.info(f"Finetune:{args.model_id}:Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logging.info(f"Finetune:{args.model_id}:Total optimization steps = {args.max_train_steps}")

    global_step = 0
    first_epoch = 0
    resume_step = 0
    if args.resume_from_checkpoint:
        path = os.path.basename(args.resume_from_checkpoint)

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            # accelerator.print(f"Resuming from checkpoint {path}")
            # accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)

    progress_bar = tqdm(range(resume_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    for epoch in range(first_epoch, args.num_train_epochs):
        logging.info(f"Finetune:{args.model_id}:Starting Epoch - {epoch+1}")
        unet.train()
        if args.modifier_tokens or args.train_text_encoder:
            text_encoder.train()
        for batch in train_dataloader:
            with accelerator.accumulate(unet):
                latents = 0.18215 * vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()

                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=latents.device).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                # Chunk the noise and model_pred into two parts and compute the loss on each part separately.
                model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
                target, target_prior = torch.chunk(target, 2, dim=0)
                # Compute instance loss
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                # Compute prior loss
                prior_loss = F.mse_loss(model_pred_prior.float(), target_prior.float(), reduction="mean")

                # Add the prior loss to the instance loss.
                loss = loss + args.prior_loss_weight * prior_loss

                accelerator.backward(loss)

                # Zero out the gradients for all token embeddings except the newly added
                # embeddings for the concept, as we only want to optimize the concept embeddings
                if args.modifier_tokens:
                    if accelerator.num_processes > 1:
                        grads_text_encoder = text_encoder.module.get_input_embeddings().weight.grad
                    else:
                        grads_text_encoder = text_encoder.get_input_embeddings().weight.grad
                    # Get the index for tokens that we want to zero the grads for
                    index_grads_to_zero = torch.arange(len(tokenizer)) != modifier_token_ids[0]
                    for i in range(len(modifier_token_ids[1:])):
                        index_grads_to_zero = index_grads_to_zero | torch.arange(len(tokenizer)) != modifier_token_ids[i]
                    grads_text_encoder.data[index_grads_to_zero, :] = grads_text_encoder.data[index_grads_to_zero, :].fill_(0)

                if accelerator.sync_gradients:
                    params_to_clip = (
                        itertools.chain(unet.parameters(), text_encoder.parameters())
                        if (args.modifier_tokens or args.train_text_encoder)
                        else unet.parameters()
                    )
                    grad_norm = accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        save_weights(global_step)
                    if global_step % 100 == 0:
                        unet.eval()
                        text_encoder.eval()
                        for module in unet.modules():
                            if hasattr(module, "convert_weight_back"):
                                module.converted_weight_check = False
                        for module in text_encoder.modules():
                            print("Setting text encoder weights back.")
                            if hasattr(module, "convert_weight_back"):
                                module.converted_weight_check = False
                        print("Logging images")
                        # prompts = ["an illustration of a dog sks on the moon", "a beautiful oil painting of dog sks", "dog sks as a human", "photo of dog sks sitting in the kindergarten", "a photo of a dog sks", "a photo of sks", "a photo of dog sks sitting in a bucket", "a dog sks sitting next to another dog sks", "Apicture of a dog sks on the moon"]
                        prompts = ["an illustration of a dog <new1> on the moon", "a beautiful oil painting of dog <new1>", "dog <new1> as a human", "photo of dog <new1> sitting in the kindergarten", "a photo of a dog <new1>", "a photo of <new1>", "a photo of dog <new1> sitting in a bucket", "dog <new1> sitting next to another dog <new1>"]
                        # prompts = ["artwork <new1> on the moon", "oil painting artwork <new1>", "dog artwork <new1>", "human artwork <new1>", "artwork <new1> of houses", "artwork <new1> of the earth", "artwork <new1> kangoroo", "abstract artwork <new1>"]
                        # prompts = ["an illustration of a piccolo <new1> on the moon", "a beautiful oil painting of piccolo <new1>", "piccolo <new1> as a human", "photo of piccolo <new1> sitting in the kindergarten", "a photo of a piccolo <new1>", "a photo of <new1>", "a photo of piccolo <new1> sitting in a bucket", "piccolo <new1> sitting next to another piccolo <new1>", "piccolo <new1> fully body pose"]
                        # prompts = ["a sketch of the eiffel tower in the style of <new1>", "a beautiful drawing in the style of <new1>", "a cute cat in the style of <new1>", "an black and white illustration in the style of <new1>", "a drawing of the earth in the style of <new1>", "an illustration of a tree in the style of <new1>"]
                        # print(batch["prompts"])
                        encoder_hidden_states_test = text_encoder(tokenizer.pad({"input_ids": tokenizer(prompts, padding="do_not_pad", truncation=True, max_length=tokenizer.model_max_length).input_ids}, padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt").input_ids.cuda())[0]
                        encoder_hidden_states_test_uncond = text_encoder(tokenizer.pad({"input_ids": tokenizer([""]*len(prompts), padding="do_not_pad", truncation=True, max_length=tokenizer.model_max_length).input_ids}, padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt").input_ids.cuda())[0]
                        sampled_latents, _ = sample(unet, noise_scheduler, encoder_hidden_states_test, batch_size=len(prompts), unconditional_conditioning=encoder_hidden_states_test_uncond, unconditional_guidance_scale=6.)
                        sampled_latents = 1 / 0.18215 * sampled_latents
                        sampled_images = vae.decode(sampled_latents.half()).sample
                        sampled_images = (sampled_images / 2 + 0.5).clamp(0, 1)
                        torchvision.utils.save_image(sampled_images, f"{global_step}.png")
                        unet.train()
                        text_encoder.train()

            progress_bar.set_postfix(loss=loss.detach().item(), prior_loss=prior_loss.detach().item(), lr=lr_scheduler.get_last_lr()[0], lr_1d= lr_scheduler.get_last_lr()[1], grad_norm=grad_norm.item())

            if global_step >= args.max_train_steps:
                break
        
        accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        save_weights(global_step, save_path=args.output_dir) 
        logging.info(f"Finetune:{args.model_id}:Finished Training & Saving embeddings")

    accelerator.end_training()

if __name__ == "__main__":
    args = parse_args()

    keyword = "dog"
    V_keyword = "<new1>"
    regularization_category = "a photo of a dog"
    concept_list_json = []
    concept_ = {}
    concept_["keyword"] = keyword
    concept_["identifier"] = V_keyword
    concept_["regularization_category"] = regularization_category
    # concept_["instance_prompt"] = f"{V_keyword} {keyword}"
    concept_["instance_prompt"] = f"dog {V_keyword}"
    concept_["class_prompt"] = regularization_category
    concept_["instance_data_dir"] = f"training_data/{keyword}/"
    concept_["class_data_dir"] = f"regularization_data/{keyword}/"
    concept_list_json.append(concept_)
    with open("concept_list.json", "w") as f:
        json.dump(concept_list_json, f)
    modifier_token = [c["identifier"] for c in concept_list_json]

    args.user_id = 1
    args.model_id = "cool"
    args.topic_id = "nice"
    args.pretrained_model_name_or_path = "runwayml/stable-diffusion-v1-5"
    args.concepts_list = "concept_list.json"
    args.prior_loss_weight = 1.0
    args.resolution = 512
    args.train_batch_size = 1
    args.learning_rate = 1e-3  # 5e-3
    args.lr_warmup_steps = 0
    args.max_train_steps = 20000
    args.num_class_images = 100
    args.scale_lr = False
    args.hflip = True
    args.modifier_tokens = modifier_token
    args.mixed_precision = "fp16"
    args.enable_xformers_memory_efficient_attention = True
    args.enable_tomesd = False
    args.train_text_encoder = True
    args.checkpointing_steps = 200
    # args.resume_from_checkpoint = ""
    # args.resume_from_checkpoint = False

    main(args)


    """
    Observations:
    Style tuning maybe lower lr (< 5e-3) -> 5e-5 seems to work on 'drawings'
    Object tuning maybe higher lr (~1e-5)
    """