import imp
import torch 
from torch._six import inf
import torch.distributed as dist

import gc
import psutil
import numpy as np
import random

import time

def mobius_logger(str):
    print(f"\033[34m[{time.asctime( time.localtime(time.time()) )}]\033[0m", end="" )
    print(f"\033[32m[Mobius Log]\033[0m {str}")
    pass

if hasattr(torch.cuda, "memory_reserved"):
    torch_memory_reserved = torch.cuda.memory_reserved
else:
    torch_memory_reserved = torch.cuda.memory_allocated
if hasattr(torch.cuda, "max_memory_reserved"):
    torch_max_memory_reserved = torch.cuda.max_memory_reserved
else:
    torch_max_memory_reserved = torch.cuda.memory_cached

def print_memory_usage(message, force=False):
    
    if force == False:
        return 

    # python doesn't do real-time garbage collection so do it explicitly to get the correct RAM reports
    gc.collect()

    # Print message except when distributed but not rank 0
    print(message)
    print(
        f"Memory Allocated {round(torch.cuda.memory_allocated() / (1024 * 1024 * 1024),2 )} GB \
        Max Memory Allocated {round(torch.cuda.max_memory_allocated() / (1024 * 1024 * 1024),2)} GB \
        Memory Reserved {round(torch_memory_reserved() / (1024 * 1024 * 1024),2)} GB \
        Max Memory Reserved {round(torch_max_memory_reserved() / (1024 * 1024 * 1024))} GB ")

    vm_stats = psutil.virtual_memory()
    used_GB = round(((vm_stats.total - vm_stats.available) / (1024**3)), 2)
    print(
        f'CPU Virtual Memory:  used = {used_GB} GB, percent = {vm_stats.percent}%')

    print("")

    # get the peak memory to report correct data, so reset the counter for the next call
    if hasattr(torch.cuda, "reset_peak_memory_stats"):  # pytorch 1.4+
        torch.cuda.reset_peak_memory_stats()


def model_config(model_name):
    """
    generate the model config according to the model name.
    """
    if model_name == "Bert":
        # 0.11B
        HIDDEN_DIM = 768
        SEQ_LEN = 512
        NUM_LAYER = 6
        NUM_HEAD = 12
    elif model_name == "Bertlarge":
        # 0.35B
        HIDDEN_DIM = 1024
        SEQ_LEN = 512
        NUM_LAYER = 24
        NUM_HEAD = 16
    elif model_name == "GPT2small":
        # 0.7B
        HIDDEN_DIM = 1536
        SEQ_LEN = 128
        NUM_LAYER = 24
        NUM_HEAD = 16
    elif model_name == "GPT2_1B":
        # 0.9B
        HIDDEN_DIM = 2048
        SEQ_LEN = 1024
        NUM_LAYER = 20
        NUM_HEAD = 16
    elif model_name == "megatron_1.3B":
        HIDDEN_DIM = 2048
        SEQ_LEN = 1024
        NUM_LAYER = 24
        NUM_HEAD = 32
    elif model_name == "GPT2_2B":
        # zero-offload
        HIDDEN_DIM = 2048
        SEQ_LEN = 1024
        NUM_LAYER = 40
        NUM_HEAD = 16
    elif model_name == "megatron_3.9B":
        # Table 4 in Megatron Paper
        HIDDEN_DIM = 2560
        SEQ_LEN = 1024
        NUM_LAYER = 24
        NUM_HEAD = 40
    elif model_name == "GPT2_4B":
        HIDDEN_DIM = 2304  # 2048
        SEQ_LEN = 1024
        NUM_LAYER = 64
        NUM_HEAD = 16
    elif model_name == "GPT3_6B":
        # 6.7B model
        HIDDEN_DIM = 3072
        SEQ_LEN = 1024
        NUM_LAYER = 53
        NUM_HEAD = 16
    elif model_name == "GPT3_8B":
        # 6.7B model
        HIDDEN_DIM = 3072
        SEQ_LEN = 1024
        NUM_LAYER = 72
        NUM_HEAD = 16
    elif model_name == "GPT3_10B":
        HIDDEN_DIM = 4096
        SEQ_LEN = 1024
        NUM_LAYER = 50
        NUM_HEAD = 16
    elif model_name == "GPT3_11B":
        HIDDEN_DIM = 4096
        SEQ_LEN = 1024
        NUM_LAYER = 55
        NUM_HEAD = 16
    elif model_name == "GPT3_12B":
        HIDDEN_DIM = 4096
        SEQ_LEN = 1024
        NUM_LAYER = 60
        NUM_HEAD = 16
    elif model_name == "GPT3_13B":
        HIDDEN_DIM = 4096
        SEQ_LEN = 1024
        NUM_LAYER = 65
        NUM_HEAD = 16
    elif model_name == "GPT3_15B":
        HIDDEN_DIM = 4096
        SEQ_LEN = 1024
        NUM_LAYER = 78
        NUM_HEAD = 16
    elif model_name == "GPT3_18B":
        HIDDEN_DIM = 4096
        SEQ_LEN = 1024
        NUM_LAYER = 90
        NUM_HEAD = 16
    # The following configs comes from paper
    # Efficient Large-Scale Language Model Training on GPU Clusters
    # NV model is wider in hidden-size
    elif model_name == "GPT_NV_18B":
        HIDDEN_DIM = 6144
        SEQ_LEN = 1024
        NUM_LAYER = 40
        NUM_HEAD = 16
    elif model_name == "GPT_NV_39B":
        HIDDEN_DIM = 8192
        SEQ_LEN = 1024
        NUM_LAYER = 48
        NUM_HEAD = 16
    elif model_name == "GPT_NV_76B":
        HIDDEN_DIM = 10240
        SEQ_LEN = 1024
        NUM_LAYER = 60
        NUM_HEAD = 16
    # The following configs comes from Deep-Offload
    # http://pasalabs.org/papers/2021/ATC21_zero-offload.pdf
    elif model_name == "GPT_DS_20B":
        HIDDEN_DIM = 8192
        SEQ_LEN = 1024
        NUM_LAYER = 25
        NUM_HEAD = 16
    elif model_name == "GPT_DS_40B":
        HIDDEN_DIM = 8192
        SEQ_LEN = 1024
        NUM_LAYER = 50
        NUM_HEAD = 16
    elif model_name == "GPT_DS_50B":
        HIDDEN_DIM = 8192
        SEQ_LEN = 1024
        NUM_LAYER = 62
        NUM_HEAD = 16
    elif model_name == "GPT_DS_60B":
        HIDDEN_DIM = 8192
        SEQ_LEN = 1024
        NUM_LAYER = 75
        NUM_HEAD = 16
    elif model_name == "GPT_DS_68B":
        HIDDEN_DIM = 9216
        SEQ_LEN = 1024
        NUM_LAYER = 66
        NUM_HEAD = 16
    # OpenAI GPT3
    elif model_name == "GPT_175B":
        HIDDEN_DIM = 12288
        SEQ_LEN = 1024
        NUM_LAYER = 96
        NUM_HEAD = 96
    elif model_name == "GPT_220B":
        HIDDEN_DIM = 12288
        SEQ_LEN = 1024
        NUM_LAYER = 120
        NUM_HEAD = 96
    elif model_name == "GPT_250B":
        HIDDEN_DIM = 12288
        SEQ_LEN = 1024
        NUM_LAYER = 137
        NUM_HEAD = 96
    elif model_name == "GPT_310B":
        HIDDEN_DIM = 16384
        SEQ_LEN = 1024
        NUM_LAYER = 128
        NUM_HEAD = 128
    elif model_name == "GPT_454B":
        HIDDEN_DIM = 20480
        SEQ_LEN = 1024
        NUM_LAYER = 90  # 105 for 530B
        NUM_HEAD = 128
    else:
        raise RuntimeError(f"The model name {model_name} is not valid!")
    assert HIDDEN_DIM % NUM_HEAD == 0
    return (HIDDEN_DIM, SEQ_LEN, NUM_LAYER, NUM_HEAD)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def debug_condition(rank=0):
    return dist.get_rank() == rank



def clip_grad_norm_(parameters, max_norm, norm_type=2):
    """Clips gradient norm of an iterable of parameters.

    This has been adapted from Nvidia megatron. We add norm averaging
    to consider MoE params when calculating norm as they will result
    in different norms across different ranks.

    This is adapted from torch.nn.utils.clip_grad.clip_grad_norm_ and
    added functionality to handle model parallel parameters. Note that
    the gradients are modified in place.

    Arguments:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.

    Returns:
        Total norm of the parameters (viewed as a single vector).
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        norms = [p.grad.detach().abs().max().to(device) for p in parameters]
        total_norm = norms[0] if len(norms) == 1 else torch.max(torch.stack(norms))
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type, dtype=torch.float32).to(device) for p in parameters]), norm_type)
    
    total_norm = total_norm.to(torch.float16)
    clip_coef = max_norm / (total_norm + 1e-6)

    if clip_coef < 1:
        for p in parameters:
            p.grad.detach().mul_(clip_coef.to(p.grad.device))
    return total_norm

