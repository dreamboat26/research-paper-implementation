import os
import torch
import random
import inspect
import logging
import accelerate
from pathlib import Path
from typing import Tuple
from PIL import Image, ImageOps
from torchvision import transforms
from torch.utils.data import Dataset
from diffusers.models.attention import CrossAttention
from ldm.models.diffusion.ddim import DDIMSampler
from accelerate.utils import set_module_tensor_to_device
from diffusers import LMSDiscreteScheduler, DDIMScheduler, PNDMScheduler, DPMSolverMultistepScheduler, EulerDiscreteScheduler, EulerAncestralDiscreteScheduler
from diffusers import UNet2DConditionModel
from safetensors.torch import safe_open
from transformers import CLIPTextModel, CLIPTextConfig
from svdiff_pytorch import UNet2DConditionModelForSVDiff, CLIPTextModelForSVDiff

SD_IMAGE_SIZE = (512, 512)

def load_unet_for_svdiff(pretrained_model_name_or_path, spectral_shifts_ckpt=None, hf_hub_kwargs=None, **kwargs):
    """
    https://github.com/huggingface/diffusers/blob/v0.14.0/src/diffusers/models/modeling_utils.py#L541
    """
    config = UNet2DConditionModel.load_config(pretrained_model_name_or_path, **kwargs)
    original_model = UNet2DConditionModel.from_pretrained(pretrained_model_name_or_path, **kwargs)
    state_dict = original_model.state_dict()
    with accelerate.init_empty_weights():
        model = UNet2DConditionModelForSVDiff.from_config(config)
    # load pre-trained weights
    param_device = "cpu"
    torch_dtype = kwargs["torch_dtype"] if "torch_dtype" in kwargs else None
    spectral_shifts_weights = {n: torch.zeros(p.shape) for n, p in model.named_parameters() if "delta" in n}
    state_dict.update(spectral_shifts_weights)
    # move the params from meta device to cpu
    missing_keys = set(model.state_dict().keys()) - set(state_dict.keys())
    if len(missing_keys) > 0:
        raise ValueError(
            f"Cannot load {model.__class__.__name__} from {pretrained_model_name_or_path} because the following keys are"
            f" missing: \n {', '.join(missing_keys)}. \n Please make sure to pass"
            " `low_cpu_mem_usage=False` and `device_map=None` if you want to randomely initialize"
            " those weights or else make sure your checkpoint file is correct."
        )

    for param_name, param in state_dict.items():
        accepts_dtype = "dtype" in set(inspect.signature(set_module_tensor_to_device).parameters.keys())
        if accepts_dtype:
            set_module_tensor_to_device(model, param_name, param_device, value=param, dtype=torch_dtype)
        else:
            set_module_tensor_to_device(model, param_name, param_device, value=param)
    
    if spectral_shifts_ckpt:
        if os.path.isdir(spectral_shifts_ckpt):
            spectral_shifts_ckpt = os.path.join(spectral_shifts_ckpt, "spectral_shifts.safetensors")
        assert os.path.exists(spectral_shifts_ckpt)

        with safe_open(spectral_shifts_ckpt, framework="pt", device="cpu") as f:
            for key in f.keys():
                # spectral_shifts_weights[key] = f.get_tensor(key)
                accepts_dtype = "dtype" in set(inspect.signature(set_module_tensor_to_device).parameters.keys())
                if accepts_dtype:
                    set_module_tensor_to_device(model, key, param_device, value=f.get_tensor(key), dtype=torch_dtype)
                else:
                    set_module_tensor_to_device(model, key, param_device, value=f.get_tensor(key))
        print(f"Resumed from {spectral_shifts_ckpt}")
    if "torch_dtype"in kwargs:
        model = model.to(kwargs["torch_dtype"])
    model.register_to_config(_name_or_path=pretrained_model_name_or_path)
    # Set model in evaluation mode to deactivate DropOut modules by default
    model.eval()
    del original_model
    torch.cuda.empty_cache()
    return model



def load_text_encoder_for_svdiff(
        pretrained_model_name_or_path,
        spectral_shifts_ckpt=None,
        **kwargs
    ):
    """
    https://github.com/huggingface/diffusers/blob/v0.14.0/src/diffusers/models/modeling_utils.py#L541
    """
    config = CLIPTextConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
    original_model = CLIPTextModel.from_pretrained(pretrained_model_name_or_path, **kwargs)
    state_dict = original_model.state_dict()
    with accelerate.init_empty_weights():
        model = CLIPTextModelForSVDiff(config)
    # load pre-trained weights
    param_device = "cpu"
    trained_token_embeds = None
    torch_dtype = kwargs["torch_dtype"] if "torch_dtype" in kwargs else None
    spectral_shifts_weights = {n: torch.zeros(p.shape) for n, p in model.named_parameters() if "delta" in n}
    state_dict.update(spectral_shifts_weights)
    # move the params from meta device to cpu
    missing_keys = set(model.state_dict().keys()) - set(state_dict.keys())
    if len(missing_keys) > 0:
        raise ValueError(
            f"Cannot load {model.__class__.__name__} from {pretrained_model_name_or_path} because the following keys are"
            f" missing: \n {', '.join(missing_keys)}. \n Please make sure to pass"
            " `low_cpu_mem_usage=False` and `device_map=None` if you want to randomely initialize"
            " those weights or else make sure your checkpoint file is correct."
        )

    for param_name, param in state_dict.items():
        accepts_dtype = "dtype" in set(inspect.signature(set_module_tensor_to_device).parameters.keys())
        if accepts_dtype:
            set_module_tensor_to_device(model, param_name, param_device, value=param, dtype=torch_dtype)
        else:
            set_module_tensor_to_device(model, param_name, param_device, value=param)
    
    if spectral_shifts_ckpt:
        if os.path.isdir(spectral_shifts_ckpt):
            spectral_shifts_ckpt = os.path.join(spectral_shifts_ckpt, "spectral_shifts_te.safetensors")
        # load state dict only if `spectral_shifts_te.safetensors` exists
        if os.path.exists(spectral_shifts_ckpt):
            with safe_open(spectral_shifts_ckpt, framework="pt", device="cpu") as f:
                trained_token_embeds = {}
                for key in f.keys():
                    if key.startswith("modifier_tokens"):
                        print(f"Adding {key} to trained_token_embeds")
                        trained_token_embeds[key.split(".")[1]] = f.get_tensor(key)
                        continue
                    # spectral_shifts_weights[key] = f.get_tensor(key)
                    accepts_dtype = "dtype" in set(inspect.signature(set_module_tensor_to_device).parameters.keys())
                    if accepts_dtype:
                        set_module_tensor_to_device(model, key, param_device, value=f.get_tensor(key), dtype=torch_dtype)
                    else:
                        set_module_tensor_to_device(model, key, param_device, value=f.get_tensor(key))
            print(f"Resumed from {spectral_shifts_ckpt}")

            # modifier_token_ids = []
            # if modifier_tokens:
            #     for modifier_token in modifier_tokens:
            #         # Add the placeholder token in tokenizer
            #         num_added_tokens = tokenizer.add_tokens(modifier_token)
            #         if num_added_tokens == 0:
            #             logging.error(f"Finetune:{args.model_id}:Token '{modifier_token}' already exists in tokenizer. Use a different one.")
            #             raise ValueError(f"Token '{modifier_token}' already exists in tokenizer. Use a different one.")

            #         modifier_token_ids.append(tokenizer.convert_tokens_to_ids(modifier_token))

            #     # Resize the token embeddings as we are adding new special tokens to the tokenizer
            #     model.resize_token_embeddings(len(tokenizer))
            #     token_embeds = model.get_input_embeddings().weight.data
            #     for modifier_token, modifier_token_id in zip(modifier_tokens, modifier_token_ids):
            #         if modifier_token in trained_token_embeds.keys():
            #             token_embeds[modifier_token_id] = trained_token_embeds[modifier_token]
            #             logging.info(f"Finetune:{args.model_id}:Loaded {modifier_token_id} token embedding from previous session.")
            #             print(f"Finetune:{args.model_id}:Loaded {modifier_token_id} token embedding from previous session.")
        
    if "torch_dtype"in kwargs:
        model = model.to(kwargs["torch_dtype"])
    # model.register_to_config(_name_or_path=pretrained_model_name_or_path)
    # Set model in evaluation mode to deactivate DropOut modules by default
    model.eval()
    del original_model
    torch.cuda.empty_cache()
    return model, trained_token_embeds

SCHEDULER_MAPPING = {
    "ddim": DDIMScheduler,
    "plms": PNDMScheduler,
    "lms": LMSDiscreteScheduler,
    "euler": EulerDiscreteScheduler,
    "euler_ancestral": EulerAncestralDiscreteScheduler,
    "dpm_solver++": DPMSolverMultistepScheduler,
}


def format_image(image):
    square_size = SD_IMAGE_SIZE
    image = Image.open(image)
    has_transparent = has_transparency(image)
    logging.info("Image has transparency")
    if has_transparent:
        logging.info("Removing transparency")
        image = remove_transparency(image, bg_colour=(34,34,34))
    logging.info("Making image square")
    image = make_image_square(image, square_size, bg_colour=(34,34,34))
    return image


def remove_transparency(image, bg_colour=(255, 255, 255)):
    
    # Only process if image has transparency (http://stackoverflow.com/a/1963146)
    if image.mode in ('RGBA', 'LA') or (image.mode == 'P' and 'transparency' in image.info):

        # # Need to convert to RGBA if LA format due to a bug in PIL (http://stackoverflow.com/a/1963146)
        # alpha = image.convert('RGBA').split()[-1]

        # # Create a new background image of our matt color.
        # # Must be RGBA because paste requires both images have the same format
        # # (http://stackoverflow.com/a/8720632  and  http://stackoverflow.com/a/9459208)
        # bg = Image.new("RGBA", image.size, bg_colour + (255,))
        # bg.paste(image, mask=alpha)
        image.load()
        bg = Image.new("RGB", image.size, bg_colour)
        bg.paste(image, mask=image.split()[3])
        image = bg
    return image

def has_transparency(image):
    if image.info.get("transparency", None) is not None:
        return True
    if image.mode == "P":
        transparent = image.info.get("transparency", -1)
        for _, index in image.getcolors():
            if index == transparent:
                return True
    elif image.mode == "RGBA":
        extrema = image.getextrema()
        if extrema[3][0] < 255:
            return True

    return False


def make_image_square(image, size, bg_colour=(255, 255, 255)):
    contained = ImageOps.contain(image, size)
    square_image = ImageOps.pad(contained, size, color=bg_colour)
    return square_image


def pad_resize_image(image: Image.Image, size: Tuple[int, int] = (512, 512), bg_color_rgb: Tuple[int, int, int] = (34, 34, 34)) -> Image.Image:
    # pads the image to a square and resizes it to the given size, removing the alpha channel in the process
    
    width, height = image.size
    max_dim = max(width, height)
    square_image = Image.new('RGBA', (max_dim, max_dim), color=bg_color_rgb + (255,))
    paste_position = ((max_dim - width) // 2, (max_dim - height) // 2)
    square_image.paste(image, paste_position)
    
    resized_image = square_image.resize(size, Image.ANTIALIAS)
    final_image = Image.new('RGB', size, bg_color_rgb)
    final_image.paste(resized_image, mask=resized_image.split()[3])

    return final_image


class DreamBoothDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(self, concepts_list, tokenizer, num_class_images=100, size=512, center_crop=False):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer

        self.instance_images_path = []
        self.class_images_path = []
        for concept in concepts_list:
            inst_img_path = [(x, concept["instance_prompt"]) for x in Path(concept["instance_data_dir"]).iterdir() if x.is_file()]
            self.instance_images_path.extend(inst_img_path)

            class_images_path = list(Path(os.path.join(concept["class_data_dir"], "images")).iterdir())
            # class_prompt = [concept["class_prompt"] for _ in range(len(class_images_path))]
            class_prompt = open(os.path.join(concept["class_data_dir"], "captions.txt")).read().split("\n")

            class_img_path = [(x, y) for (x,y) in zip(class_images_path,class_prompt)]
            self.class_images_path.extend(class_img_path[:num_class_images])

        random.shuffle(self.instance_images_path)
        self.num_instance_images = len(self.instance_images_path)
        self.num_class_images = len(self.class_images_path)
        self._length = max(self.num_class_images, self.num_instance_images)
        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        instance_image, instance_prompt = self.instance_images_path[index % self.num_instance_images]
        instance_image = Image.open(instance_image)
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        example["instance_images"] = self.image_transforms(instance_image)
        example["instance_prompt_ids"] = self.tokenizer(
            instance_prompt,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids

        class_image, class_prompt = self.class_images_path[index % self.num_class_images]
        class_image = Image.open(class_image)
        if not class_image.mode == "RGB":
            class_image = class_image.convert("RGB")
        example["class_images"] = self.image_transforms(class_image)
        example["class_prompt_ids"] = self.tokenizer(
            class_prompt,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids

        return example

def collate_fn(examples, with_prior_preservation=False):
    input_ids = [example["instance_prompt_ids"] for example in examples]
    pixel_values = [example["instance_images"] for example in examples]

    # Concat class and instance examples for prior preservation.
    # We do this to avoid doing two forward passes.
    if with_prior_preservation:
        input_ids += [example["class_prompt_ids"] for example in examples]
        pixel_values += [example["class_images"] for example in examples]

    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    input_ids = torch.cat(input_ids, dim=0)

    batch = {
        "input_ids": input_ids,
        "pixel_values": pixel_values,
    }
    return batch

def freeze_parameters(parameters, receive_gradients=False):
    """
    Takes in model parameters, iterates through them and either freezes them or unfreezes them.
    """
    for p in parameters:
        p.requires_grad = receive_gradients


def new_forward(self, hidden_states, context=None):
        _, sequence_length, _ = hidden_states.shape
        crossattn = False
        if context is not None:
            crossattn = True

        query = self.to_q(hidden_states)
        context = context if context is not None else hidden_states
        key = self.to_k(context)
        value = self.to_v(context)

        if crossattn:
            modifier = torch.ones_like(key)
            modifier[:, :1, :] = modifier[:, :1, :]*0.
            key = modifier*key + (1-modifier)*key.detach()
            value = modifier*value + (1-modifier)*value.detach()

        dim = query.shape[-1]

        query = self.reshape_heads_to_batch_dim(query)
        key = self.reshape_heads_to_batch_dim(key)
        value = self.reshape_heads_to_batch_dim(value)

        if self._slice_size is None or query.shape[0] // self._slice_size == 1:
            hidden_states = self._attention(query, key, value)
        else:
            hidden_states = self._sliced_attention(query, key, value, sequence_length, dim)

        # linear proj
        hidden_states = self.to_out[0](hidden_states)
        # dropout
        hidden_states = self.to_out[1](hidden_states)
        return hidden_states


def adjust_forward(model):
    for l in model.children():
        if isinstance(l, CrossAttention):
            setattr(l, 'forward', new_forward.__get__(l, l.__class__))
        else:
            adjust_forward(l)


def modify_unet(unet):
    for name, params in unet.named_parameters():
        if 'attn2.to_k' in name or 'attn2.to_v' in name:
            params.requires_grad = True
        else:
            params.requires_grad = False

    adjust_forward(unet)
    return unet


def save_progress(text_encoder, unet, modifier_token_id, accelerator, args, save_path):
    delta_dict = {'unet': {},}
    if args.modifier_tokens is not None:
        for i in range(len(modifier_token_id)):
            delta_dict[args.modifier_tokens[i]] = accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[modifier_token_id[i]].detach().cpu()
    for name, params in accelerator.unwrap_model(unet).named_parameters():
        if 'attn2.to_k' in name or 'attn2.to_v' in name:
            delta_dict['unet'][name] = params.cpu().clone()

    torch.save(delta_dict, save_path)


def load_model(text_encoder, tokenizer, unet, save_path):
    st = torch.load(save_path)
    if 'modifier_token' in st:
        modifier_tokens = list(st['modifier_token'].keys())
        modifier_token_id = []
        for modifier_token in modifier_tokens:
            _ = tokenizer.add_tokens(modifier_token)
            modifier_token_id.append(tokenizer.convert_tokens_to_ids(modifier_token))

        text_encoder.resize_token_embeddings(len(tokenizer))
        token_embeds = text_encoder.get_input_embeddings().weight.data
        for i, id_ in enumerate(modifier_token_id):
            token_embeds[id_] = st['modifier_token'][modifier_tokens[i]]

    for name, params in unet.named_parameters():
        if 'attn2.to_k' in name or 'attn2.to_v' in name:
            params.data.copy_(st['unet'][f'{name}'])


@torch.no_grad()
def sample(model, scheduler, cond, batch_size, ddim_steps=50, h=512, w=512, **kwargs):
    ddim_sampler = DDIMSampler(model, scheduler)
    shape = (4, h // 8, w // 8)
    samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size, shape, cond, verbose=False, **kwargs)
    return samples, intermediates
