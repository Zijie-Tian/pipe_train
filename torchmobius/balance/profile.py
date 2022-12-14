"""Per-layer profilers."""
from audioop import avg
import copy
import time
from typing import Generator, List, Tuple, Union

import torch
from torch import Tensor
import torch.nn as nn

from torchmobius.microbatch import Batch

__all__: List[str] = []


Device = Union[torch.device, int, str]

Tensors = Tuple[Tensor, ...]
TensorOrTensors = Union[Tensor, Tensors]


def layerwise_sandbox(module: nn.Sequential,
                      device: torch.device,
                      ) -> Generator[nn.Module, None, None]:
    """Copies layers for ease to profile. It doesn't modify the given
    module.
    """
    for layer in module:
        layer_copy = copy.deepcopy(layer)
        layer_copy.to(device)
        layer_copy.train()
        yield layer_copy


def detach(batch: Batch) -> None:
    """Detaches from autograd graph."""
    for i, x in enumerate(batch):
        batch[i] = x.detach().requires_grad_(x.requires_grad)

""" NOTE(Mobius)
"""
    
def mobius_profile(module: nn.Sequential,
                   sample: TensorOrTensors,
                   device: torch.device):
    if any(p.grad is not None for p in module.parameters()):
        raise ValueError('some parameter already has gradient')
    

    _batch = Batch(sample)
    for i, x in enumerate(_batch):
        _batch[i] = x.detach().to(device).requires_grad_(x.requires_grad)

    t_fwd:    List[List[float]] = [[] for _ in module]
    t_bwd:    List[List[float]] = [[] for _ in module]
    s_param:  List[List[float]] = [[] for _ in module]
    s_act:    List[List[float]] = [[] for _ in module]
    s_fwd:    List[List[float]] = [[] for _ in module]
    s_bwd:    List[List[float]] = [[] for _ in module]
    
    
    times = 0
    
    while times < 5:
        batch = _batch
        for i, layer in enumerate(layerwise_sandbox(module, device)):
            detach(batch)

            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            
            # Forward
            # clean the max memory stats
            torch.cuda.reset_accumulated_memory_stats(device)
            # memory used before the forward
            memory_before = torch.cuda.memory_allocated(device)
            # timer
            tick = time.time()
            
            with torch.no_grad():
                fwd_batch = batch.call(layer)
                
            tock = time.time()
            memory_after = torch.cuda.max_memory_allocated(device)
            
            
            if times >= 3:
                if batch.atomic:
                    s_act[i].append(batch.value.storage().size() * batch.value.storage().element_size())
                else:
                    s_act[i].append(
                        sum(p.storage().size() * p.storage().element_size()
                            for p in batch.value)
                    )
                
                # working time
                t_fwd[i].append(tock - tick)
                # working memory
                s_fwd[i].append(memory_after - memory_before)
            


            # Backward
            # clean the max memory stats
            torch.cuda.reset_accumulated_memory_stats(device)
            # memory used before the forward
            memory_before = torch.cuda.memory_allocated(device)
            # timer
            tick = time.time()
            
            # recomputation
            batch = batch.call(layer)
            backward_tensors = tuple(y for y in batch if y.requires_grad)
            if backward_tensors:
                torch.autograd.backward(backward_tensors, backward_tensors)
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
                
            tock = time.time()
            memory_after = torch.cuda.max_memory_allocated(device)
            
            if times >= 3:
                p_size = sum(p.storage().size() * p.storage().element_size()
                            for p in layer.parameters())
                s_param[i].append(p_size)
                t_bwd[i].append(tock - tick)
                s_bwd[i].append(memory_after - memory_before)
        times += 1
            
    us = 1e6
    return [sum(t * us for t in buf) / len(buf) for buf in t_fwd], \
           [sum(t * us for t in buf) / len(buf) for buf in t_bwd], \
           [sum(s for s in buf) / len(buf) for buf in s_param], \
           [sum(s for s in buf) / len(buf) for buf in s_act], \
           [sum(s for s in buf) / len(buf) for buf in s_fwd], \
           [sum(s for s in buf) / len(buf) for buf in s_bwd]



def profile_times(module: nn.Sequential,
                  sample: TensorOrTensors,
                  timeout: float,
                  device: torch.device,
                  ) -> List[int]:
    """Profiles elapsed times per layer."""
    if any(p.grad is not None for p in module.parameters()):
        raise ValueError('some parameter already has gradient')

    _batch = Batch(sample)
    for i, x in enumerate(_batch):
        _batch[i] = x.detach().to(device).requires_grad_(x.requires_grad)

    time_bufs: List[List[float]] = [[] for _ in module]
    begun_at = time.time()

    while time.time() - begun_at < timeout:
        batch = _batch
        for i, layer in enumerate(layerwise_sandbox(module, device)):
            detach(batch)

            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            tick = time.time()

            # Forward
            batch = batch.call(layer)

            # Backward
            backward_tensors = tuple(y for y in batch if y.requires_grad)
            if backward_tensors:
                torch.autograd.backward(backward_tensors, backward_tensors)

            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            tock = time.time()

            time_bufs[i].append(tock - tick)

    us = 1_000_000
    return [sum(int(t*us) for t in buf) for buf in time_bufs]


def profile_sizes(module: nn.Sequential,
                  input: TensorOrTensors,
                  chunks: int,
                  param_scale: float,
                  device: torch.device,
                  ) -> List[int]:
    """Profiles CUDA memory usage per layer."""
    if device.type != 'cuda':
        raise ValueError('size profiler supports only CUDA device')

    batch = Batch(input)
    sizes: List[int] = []

    latent_scale = batch[0].size(0) / chunks
    for i, x in enumerate(batch):
        batch[i] = x[:1].detach().to(device).requires_grad_(x.requires_grad)

    for layer in layerwise_sandbox(module, device):
        detach(batch)

        # Detect memory usage at forward.
        memory_before = torch.cuda.memory_allocated(device)
        batch = batch.call(layer)
        memory_after = torch.cuda.memory_allocated(device)
        latent_size = memory_after - memory_before

        # Analyze size of parameters.
        param_size = sum(p.storage().size() * p.storage().element_size()
                         for p in layer.parameters())

        # Combine size of parameters and activations with normalize scales.
        size = latent_size*latent_scale + param_size*param_scale
        sizes.append(int(size))

    return sizes
