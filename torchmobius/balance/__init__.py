"""A helper to roughly balance a sequential module.

Usage::

    import torch
    from torchgpipe import GPipe
    from torchgpipe.balance import balance_by_time

    sample = torch.empty(128, 3, 224, 224)
    balance = balance_by_time(torch.cuda.device_count(), model, sample)

    gpipe = GPipe(model, balance, chunks=8)

"""
from typing import List, Tuple, Union

import torch
from torch import Tensor
import torch.nn as nn

from torchmobius.balance import blockpartition
from torchmobius.balance.profile import profile_sizes, profile_times

__all__ = ['balance_by_time', 'balance_by_size']


Device = Union[torch.device, int, str]

Tensors = Tuple[Tensor, ...]
TensorOrTensors = Union[Tensor, Tensors]


def balance_cost(cost: List[int], partitions: int) -> List[int]:
    partitioned = blockpartition.solve(cost, partitions)
    return [len(p) for p in partitioned]


def balance_by_time(partitions: int,
                    module: nn.Sequential,
                    sample: TensorOrTensors,
                    *,
                    timeout: float = 1.0,
                    device: Device = torch.device('cuda'),
                    ) -> List[int]:
    """Naive automatic balancing by elapsed time per layer.
    ::

        sample = torch.empty(128, 3, 224, 224)
        balance = balance_by_time(torch.cuda.device_count(), model, sample)
        gpipe = GPipe(model, balance, chunks=8)

    Args:
        partitions (int):
            intended number of partitions
        module (torch.nn.Sequential):
            sequential module to be partitioned
        sample (torch.Tensor):
            example input with arbitrary batch size

    Keyword Args:
        timeout (float):
            profiling iterates again if the timeout (in second) is not exceeded
            (default: ``1.0``)
        device ('cpu' or 'cuda' device):
            CPU or CUDA device where each layer is profiled (default: the
            current CUDA device)

    Returns:
        A list of number of layers in each partition. Use it for the `balance`
        parameter of :class:`~torchgpipe.GPipe`.

    .. note::
        `module` and `sample` must be placed on the same device.

    """
    times = profile_times(module, sample, timeout, torch.device(device))
    return balance_cost(times, partitions)


def balance_by_size(partitions: int,
                    module: nn.Sequential,
                    input: TensorOrTensors,
                    *,
                    chunks: int = 1,
                    param_scale: float = 2.0,
                    device: Device = torch.device('cuda'),
                    ) -> List[int]:
    """Naive automatic balancing by CUDA memory usage per layer.

    During training, required memory for parameters depends on which optimizer
    is used. Optimizers may use buffers for each parameter to track
    optimization statistics internally, such as momentum buffer in SGD.

    To get more reliable size based balance, you should specify `param_scale`
    with regard to your optimizer. The default `param_scale` is 2 instead of 1
    due to gradient accumulation which is necessary for every optimizer.

    Follow this guide to choose correct `param_scale` for typical optimizers:

    =========  =============  =========================================
    Optimizer  `param_scale`  Internal State
    =========  =============  =========================================
    SGD        2--3           (momentum_buffer)
    Adam       4--5           exp_avg, exp_avg_sq, (max_exp_avg_sq)
    Adadelta   4              square_avg, acc_delta
    Adagrad    3              sum
    RMSprop    3--5           square_avg, (momentum_buffer), (grad_avg)
    =========  =============  =========================================

    Here's a simple example with the Adam optimizer::

        balance = balance_by_size(
            torch.cuda.device_count(),
            model,

            # Same size with mini-batch to train
            torch.empty(1024, 3, 224, 224),

            # Number of micro-batches to train with GPipe
            chunks=8,

            # 4 for Adam
            param_scale=4.0,
        )

        gpipe = GPipe(model, balance, chunks=8)
        adam = Adam(gpipe.parameters())

    Args:
        partitions (int):
            intended number of partitions
        module (torch.nn.Sequential):
            sequential module to be partitioned
        input (torch.Tensor):
            example mini-batch with the same size to train

    Keyword Args:
        chunks (int):
            number of micro-batches will be used to train (default: ``1``)
        param_scale (float):
            how many copies of parameters would be allocated for training. It
            depends on optimizer. See the above guide. (default: ``2.0``)
        device ('cuda' device):
            CUDA device where each layer is profiled (default: the current CUDA
            device)

    Returns:
        A list of number of layers in each partition. Use it for the `balance`
        parameter of :class:`~torchgpipe.GPipe`.

    .. note::
        `module` and `input` must be placed on the same CUDA device.

    """
    sizes = profile_sizes(module, input, chunks, param_scale, torch.device(device))
    return balance_cost(sizes, partitions)
