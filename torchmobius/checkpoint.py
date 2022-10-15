"""Checkpointing with preceding recomputation.

PyTorch already provides the official checkpointing utilities in
:mod:`torch.utils.checkpoint`. The official checkpointing combines
recomputation and recursive backpropagation into one autograd function named
``CheckpointFunction``. Hence, the recomputation can be started only when the
gradients arrive to the function. In GPipe, the recomputation needs to precede
the gradient arrival to minimize the GPU idle time.

We solve this problem by introducing separate autograd functions named
:class:`Recompute` and :class:`Checkpoint`. Each function represents
recomputation and recursive backpropagation, respectively. We can manipulate
the control flow in aspect of both the autograd engine and CUDA with a pair of
the functions.

Specifically, we place CUDA stream synchronization between :class:`Recompute`
and :class:`Checkpoint` to delay only :class:`Checkpoint` until the gradient is
copied entirely.

"""
from ast import excepthandler
from collections import deque
from contextlib import contextmanager
import threading
from typing import TYPE_CHECKING, Deque, Generator, List, Optional, Tuple, Union

import torch
from torch import ByteTensor, Tensor
import torch.autograd

from torchmobius.dependency import fork, join
from torchmobius.microbatch import Batch
from torchmobius.phony import get_phony
from torchmobius.attribute import MobiusPosistion

__all__ = ['is_checkpointing', 'is_recomputing']


Tensors = Tuple[Tensor, ...]
TensorOrTensors = Union[Tensor, Tensors]

# Types for shared memory between Checkpoint and Recompute.
Recomputed = Tuple[TensorOrTensors, Tensors]         # (output, input_leaf)
RNGStates = Tuple[ByteTensor, Optional[ByteTensor]]  # (cpu_rng_state, gpu_rng_state)


if TYPE_CHECKING:
    from typing_extensions import Protocol
else:
    Protocol = object


# Protocol with __call__ instead of Callable can be used as an attribute type.
# See: https://github.com/python/mypy/issues/708#issuecomment-561735949
class Function(Protocol):
    def __call__(self, input: TensorOrTensors) -> TensorOrTensors:
        ...


def checkpoint(function: Function, input: TensorOrTensors) -> TensorOrTensors:
    """Makes a checkpoint with a simple interface like
    :func:`torch.utils.checkpoint.checkpoint`. It's only used to test or debug
    :class:`Checkpoint` and :class:`Recompute` without boilerplate.
    """
    batch = Batch(input)

    chk = Checkpointing(function, batch)
    batch = chk.checkpoint()
    chk.recompute(batch)

    return batch.tensor_or_tensors


class Checkpointing:
    """Generates a pair of :class:`Checkpoint` and :class:`Recompute`."""

    def __init__(self, function: Function, batch: Batch,
                 bind_module  =None,
                 transfer_time=None,
                 upload_streams_dic=None,
                 offload_streams_dic=None,
                 microbatch_parallelism=1
                 ) -> None:

        self.function = function
        self.batch = batch

        # Shared memory between Checkpoint and Recompute. 1-length deque is
        # used for mutability and length limitation.
        self.recomputed: Deque[Recomputed] = deque(maxlen=microbatch_parallelism)
        self.rng_states: Deque[RNGStates] = deque(maxlen=microbatch_parallelism)

        self.bind_module = bind_module
        self.transfer_time = transfer_time
        self.upload_streams_dic = upload_streams_dic
        self.offload_streams_dic = offload_streams_dic
        
        self.microbatch_parallelism = microbatch_parallelism
        
    def checkpoint(self) -> Batch:
        """Returns a batch applied by :class:`Checkpoint`."""
        input_atomic = self.batch.atomic
        input = tuple(self.batch)

        # Use a phony which requires grad to ensure that Checkpoint can be
        # tracked by the autograd engine even when none of the input tensors
        # require grad.
        phony = get_phony(self.batch[0].device, requires_grad=True)

        output = Checkpoint.apply(phony, self.recomputed, self.rng_states,
                                  self.function, input_atomic, 
                                  self.microbatch_parallelism, *input)
        return Batch(output)

    def recompute(self, batch) -> None:
        """Applies :class:`Recompute` to the batch in place."""

        input_atomic = self.batch.atomic
        input = tuple(self.batch)

        # batch[0] is always requiring grad, because it has been passed
        # checkpoint with a phony requiring grad.
        batch[0], phony = fork(batch[0])
        phony = Recompute.apply(phony, self.recomputed, self.rng_states,
                                self.function, input_atomic, 

                                self.bind_module,
                                self.transfer_time,
                                self.upload_streams_dic,
                                self.offload_streams_dic,
                                self.microbatch_parallelism,
                                
                                *input)
        batch[0] = join(batch[0], phony)


class ThreadLocal(threading.local):
    def __init__(self) -> None:
        self.is_checkpointing = False
        self.is_recomputing = False


thread_local = ThreadLocal()


@contextmanager
def enable_checkpointing() -> Generator[None, None, None]:
    """Makes :func:`is_checkpointing` return :data:`True` within a context."""
    orig = thread_local.is_checkpointing
    thread_local.is_checkpointing = True
    try:
        yield
    finally:
        thread_local.is_checkpointing = orig


@contextmanager
def enable_recomputing() -> Generator[None, None, None]:
    """Makes :func:`is_recomputing` return :data:`True` within a context."""
    orig = thread_local.is_recomputing
    thread_local.is_recomputing = True
    try:
        yield
    finally:
        thread_local.is_recomputing = orig


def is_checkpointing() -> bool:
    """Whether the current forward propagation is under checkpointing.

    Returns:
        bool: :data:`True` if it's under checkpointing.

    """
    return thread_local.is_checkpointing


def is_recomputing() -> bool:
    """Whether the current forward propagation is under checkpoint
    recomputation. Use this to prevent duplicated side-effects at forward
    propagation::

        class Counter(nn.Module):
            def __init__(self):
                super().__init__()
                self.counter = 0

            def forward(self, input):
                if not is_recomputing():
                    self.counter += 1
                return input

    Returns:
        bool: :data:`True` if it's under checkpoint recomputation.

    .. seealso:: :ref:`Detecting Recomputation`

    """
    return thread_local.is_recomputing


class Context:
    """The common interface between the :class:`Checkpoint` and
    :class:`Recompute` context.
    """
    recomputed: Deque[Recomputed]
    rng_states: Deque[RNGStates]
    function: Function
    input_atomic: bool

    saved_tensors: Tuple[Tensor, ...]

    def save_for_backward(self, *tensors: Tensor) -> None:  # pragma: no cover
        pass


def save_rng_states(device: torch.device,
                    rng_states: Deque[RNGStates],
                    ) -> None:
    """:meth:`Checkpoint.forward` captures the current PyTorch's random number
    generator states at CPU and GPU to reuse in :meth:`Recompute.backward`.

    .. seealso:: :ref:`Referential Transparency`

    """
    cpu_rng_state = torch.get_rng_state()

    gpu_rng_state: Optional[ByteTensor]
    if device.type == 'cuda':
        gpu_rng_state = torch.cuda.get_rng_state(device)
    else:
        gpu_rng_state = None

    rng_states.append((cpu_rng_state, gpu_rng_state))

@contextmanager
def restore_rng_states(device: torch.device,
                       rng_states: Deque[RNGStates],
                       ) -> Generator[None, None, None]:
    """:meth:`Recompute.backward` restores the random number generator states
    captured by :func:`save_rng_states` within its context.

    .. seealso:: :ref:`Referential Transparency`

    """
    cpu_rng_state, gpu_rng_state = rng_states.pop()

    gpu_devices: List[torch.device] = []
    if device.type == 'cuda':
        gpu_devices.append(device)

    with torch.random.fork_rng(gpu_devices):
        torch.set_rng_state(cpu_rng_state)
        if gpu_rng_state is not None:
            torch.cuda.set_rng_state(gpu_rng_state, device)
        yield


class Checkpoint(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Context,  # type: ignore
                phony: Tensor,
                recomputed: Deque[Recomputed],
                rng_states: Deque[RNGStates],
                function: Function,
                input_atomic: bool,
                microbatch_parallelism: int,
                *input: Tensor,
                ) -> TensorOrTensors:
        ctx.recomputed = recomputed
        ctx.rng_states = rng_states

        save_rng_states(input[0].device, ctx.rng_states)
            
        # ctx.save_for_backward(*input)

        ctx.function = function
        ctx.input_atomic = input_atomic
        
        
        with torch.no_grad(), enable_checkpointing():
            output = function(input[0] if input_atomic else input)

        return output

    @staticmethod
    def backward(ctx: Context,
                 *grad_output: Tensor,
                 ) -> Tuple[Optional[Tensor], ...]:  # pragma: no cover
        
        output, input_leaf = ctx.recomputed.pop()
            
        if isinstance(output, tuple):
            tensors = output
        else:
            tensors = (output,)
            
        if any(y.requires_grad for y in tensors):
            torch.autograd.backward(tensors, grad_output)
        
        grad_input: List[Optional[Tensor]] = [None, None, None, None, None, None]
        grad_input.extend(x.grad for x in input_leaf)
        return tuple(grad_input)


class Recompute(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Context,  # type: ignore
                phony: Tensor,
                recomputed: Deque[Recomputed],
                rng_states: Deque[RNGStates],
                function: Function,
                input_atomic: bool,
                
                bind_module=None,
                transfer_time=None,
                upload_streams_dic=None,
                offload_streams_dic=None,
                
                microbatch_parallelism=None,
                
                *input: Tensor,
                ) -> Tensor:
        ctx.recomputed = recomputed
        ctx.rng_states = rng_states

        ctx.function = function
        ctx.input_atomic = input_atomic
                
        ctx.save_for_backward(*input)

        #NOTE(Mobius) flush the input to DRAM
        # if bind_module != None:
        #     if type(input) is tuple:
        #         for i in input:
        #             bind_module.mobius_module_attr.bind_and_free_activation(
        #                 activation=i,
        #                 transfer_time=transfer_time,
        #                 upload_stream=upload_streams_dic[input[0].device],
        #                 offload_stream=offload_streams_dic[input[0].device],
        #             )

        return phony

    @staticmethod
    def backward(ctx: Context, *grad_output: Tensor) -> Tuple[None, ...]:  # pragma: no cover
        input = ctx.saved_tensors

        # NOTE(Mobius) free activation
        # for i in input:
        #     if hasattr(i, 'mobius_tensor_attr'):
        #         while True:
        #             if i.mobius_tensor_attr.get_position() == MobiusPosistion.GPU:
        #                 # # NOTE(fyy) dirty sync
        #                 if torch.isnan(i.mobius_tensor_attr.device_param_tensor.data).any():
        #                     print('nan in backward')
        #                 break
        
        input_leaf = tuple(x.detach().requires_grad_(x.requires_grad) for x in input)

        with restore_rng_states(input[0].device, ctx.rng_states):
            with torch.enable_grad(), enable_recomputing():
                output = ctx.function(input_leaf[0] if ctx.input_atomic else input_leaf)

        ctx.recomputed.append((output, input_leaf))

        
        #NOTE(Mobius) free activation
        # for i in input:
        #     if hasattr(i, 'mobius_tensor_attr'):
        #         i.mobius_tensor_attr.free_param()
        #         del i

        
                
        grad_input: List[None] = [None, None, None, None, None, None, None, None, None, None, None, None]
        grad_input.extend(None for _ in ctx.saved_tensors)
        return tuple(grad_input)
