"""Arbitrary dependency between two autograd lanes."""
from typing import List, Tuple

import torch
from torch import Tensor

from torchmobius.phony import get_phony

__all__: List[str] = []


def fork(input: Tensor) -> Tuple[Tensor, Tensor]:
    """Branches out from an autograd lane of the given tensor."""
    if torch.is_grad_enabled() and input.requires_grad:
        input, phony = Fork.apply(input)
    else:
        phony = get_phony(input.device, requires_grad=False)

    return input, phony


class Fork(torch.autograd.Function):
    @staticmethod
    def forward(ctx: 'Fork', input: Tensor) -> Tuple[Tensor, Tensor]:  # type: ignore
        phony = get_phony(input.device, requires_grad=False)
        return input.detach(), phony.detach()

    @staticmethod
    def backward(ctx: 'Fork', grad_input: Tensor, grad_grad: Tensor) -> Tensor:  # type: ignore
        return grad_input


def join(input: Tensor, phony: Tensor) -> Tensor:
    """Merges two autograd lanes."""
    if torch.is_grad_enabled() and (input.requires_grad or phony.requires_grad):
        input = Join.apply(input, phony)

    return input


class Join(torch.autograd.Function):
    @staticmethod
    def forward(ctx: 'Join', input: Tensor, phony: Tensor) -> Tensor:  # type: ignore
        return input.detach()

    @staticmethod
    def backward(ctx: 'Join', grad_input: Tensor) -> Tuple[Tensor, None]:  # type: ignore
        return grad_input, None
