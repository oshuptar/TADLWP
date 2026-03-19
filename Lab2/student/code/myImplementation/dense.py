import torch
from torch.autograd import Function
import torch.nn.functional as F
import torch.nn as nn
from typing import Optional, Tuple

class _DenseFn(Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        y = None
        raise NotImplementedError("TODO: Implement the forward pass")
        ctx.save_for_backward(x, weight, bias)
        return y

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, weight, bias = ctx.saved_tensors
        N = x.shape[0]
        grad_x = grad_w = grad_b = None
        raise NotImplementedError("TODO: Implement the backward pass - compute gradients for x, weight, and bias")
        return grad_x, grad_w, grad_b

class Dense(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias   = nn.Parameter(torch.empty(out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in = self.weight.size(1)
            bound = 1.0 / fan_in**0.5
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _DenseFn.apply(x, self.weight, self.bias)
