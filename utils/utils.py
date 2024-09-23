
import torch
import random
import numpy as np
from transformers import set_seed

import os
import torch.distributed as dist
import utils.save_ckpt as save_ckpt

def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    
    # counting params in the prompt 
    for _, param in model.named_parameters():
        num_params = param.numel()
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    # counting params in the pretrained large language model
    # for _, param in model.lm_model.named_parameters():
    #     num_params = param.numel()
    #     if num_params == 0 and hasattr(param, "ds_numel"):
    #         num_params = param.ds_numel

    #     all_param += num_params
    #     if param.requires_grad:
    #         trainable_params += num_params
    
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")

def save_checkpoint(step, epoch, model, optimizer, params):
    
    state = {
        "step": step,
        "epoch": epoch,
        "model": model,
        "optimizer": optimizer.state_dict(),
    }
    save_ckpt.save(state, params.output_dir, params.keep_checkpoint_max)

def print_rank_0(msg, rank=0):
    if rank <= 0:
        print(msg)


def to_device(batch, device):
    output = {}
    for k, v in batch.items():
        try:
            output[k] = v.to(device)
        except:
            output[k] = v
    return output


def set_random_seed(seed):
    if seed is not None:
        set_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def save_model(model, tokenizer, output_dir, model_name, state_dict=None):
    save_dir = os.path.join(output_dir, model_name)
    if state_dict == None:
        model.save_pretrained(save_dir, torch_dtype=torch.float16)
    else:
        model.save_pretrained(save_dir, state_dict=state_dict, torch_dtype=torch.float16)
    tokenizer.save_pretrained(save_dir)