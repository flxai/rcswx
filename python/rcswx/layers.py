# Derived from einsearch 7e713c7951397a6b12bd57638409774a74381746.
# Copyright (c) 2024 Adri Gómez Martín, Felix Möller, Linus Ericsson, Aaron Klein.
# Distributed under the MIT license; see LICENSE.einsearch.
"""Actual Torch constructors used by the reference grammar families."""

from math import floor, sqrt

import torch
from torch import nn
from torch.nn import functional as F


def pair(x):
    if isinstance(x, tuple) or (isinstance(x, list) and len(x) == 2):
        return x
    elif isinstance(x, int):
        return (x, x)


class Lambda(nn.Module):
    """Module that applies a lambda function to the input."""

    def __init__(self, lambd):
        super().__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

    def __repr__(self):
        return f"Lambda({self.lambd})"


class SequentialModule(nn.Module):
    """
    Module that applies a sequence of functions to the input.

    Attributes
    ----------
    first_fn : nn.Module
        The first function that is applied to the input
    second_fn : nn.Module
        The second function that is applied to the output of the first function
    """

    def __init__(self, first_fn, second_fn):
        super().__init__()
        self.first_fn = first_fn
        self.second_fn = second_fn

    def forward(self, x):
        out = self.first_fn(x)
        out = self.second_fn(out)
        return out


class SequentialModule4(nn.Module):
    """
    Module that applies a sequence of functions to the input.

    Attributes
    ----------
    fns : nn.Module
        The function that is applied to the input four times
    """

    def __init__(self, fns):
        super().__init__()
        self.fns = nn.ModuleList(fns)

    def forward(self, x):
        for fn in self.fns:
            x = fn(x)
        return x


class BranchingModule(nn.Module):
    """
    Module that branches the input and then aggregates the outputs.

    Attributes
    ----------
    branching_fn : nn.Module
        The function that branches the input (only supports a branching factor of 2 right now)
    inner_fn : nn.Module
        The list of functions that is applied to each branch, respectively (each branch is a separate module)
    aggregation_fn : nn.Module
        The function that aggregates the outputs of the inner functions (only supports a branching factor of 2 right now)
    """

    def __init__(self, branching_fn, inner_fn, aggregation_fn):
        super().__init__()
        self.branching_fn = branching_fn
        self.inner_fn = nn.ModuleList(inner_fn)
        self.aggregation_fn = aggregation_fn

    def forward(self, x):
        branching_outs = list(self.branching_fn(x))
        inner_outs = []
        for i in range(len(branching_outs)):
            inner_out = self.inner_fn[i](branching_outs[i])
            inner_outs.append(inner_out)
        aggregation_out = self.aggregation_fn(inner_outs)
        return aggregation_out


class RoutingModule(nn.Module):
    """
    Module that applies a sequence of functions to the input.

    Attributes
    ----------
    prerouting_fn : nn.Module
        The function that is applied before the computation function, rearranges/permutes the input
    computation_fn : nn.Module
        The module that processes the output of the prerouting function
    postrouting_fn : nn.Module
        The function that is applied after the computation function, rearranges/permutes the output
    """

    def __init__(self, prerouting_fn, inner_fn, postrouting_fn):
        super().__init__()
        self.prerouting_fn = prerouting_fn
        self.inner_fn = inner_fn
        self.postrouting_fn = postrouting_fn
        if hasattr(self.prerouting_fn, "fold_output_shape"):
            self.postrouting_fn.output_shape = self.prerouting_fn.fold_output_shape

    def forward(self, x):
        out = self.prerouting_fn(x)
        out = self.inner_fn(out)
        out = self.postrouting_fn(out)
        return out


class ComputationModule(nn.Module):
    """
    Module that applies a sequence of functions to the input.

    Attributes
    ----------
    computation_fn : nn.Module
        The function that is applied to the input, e.g. a linear layer or a normalization layer
    """

    def __init__(self, computation_fn):
        super().__init__()
        self.computation_fn = computation_fn

    def forward(self, x):
        out = self.computation_fn(x)
        return out


class CloneTensor(nn.Module):
    """Clone a tensor a given number of times."""

    def __init__(self, num_clones, **kwargs):
        super().__init__()
        self.n = num_clones

    def forward(self, x):
        return (torch.clone(x) for _ in range(self.n))

    def __repr__(self):
        return f"CloneTensor(n={self.n})"


class GroupDim(nn.Module):
    """Group a tensor along a given dimension. E.g split a tensor along the second dimension into 2 groups of equal size."""

    def __init__(self, splits, dim, dim_total, **kwargs):
        super().__init__()
        self.sections = [dim_total // splits] * splits
        self.n = splits
        self.dim = dim

    def forward(self, x):
        return torch.split(x, self.sections, dim=self.dim)

    def __repr__(self):
        return f"GroupDim(splits={self.n}, dim={self.dim})"


class Im2Col(nn.Module):
    """
    Rearrange the dimensions of a tensor to form a matrix.
    This is the inverse of the Col2Im class.
    It converts from the "im" mode to the "col" mode.
    """

    def __init__(
        self, input_shape, kernel_size, stride=1, padding=0, dilation=1, groups=1, **kwargs
    ):
        super().__init__()
        batch, channels, height, width = input_shape
        self.kernel_size, self.stride, self.padding, self.dilation = (
            pair(kernel_size),
            pair(stride),
            pair(padding),
            pair(dilation),
        )
        self.prearrange = lambda x: x
        self.postarrange = lambda x: x.permute(0, 2, 1)
        self.unfold = nn.Unfold(
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )
        self.fold_output_shape = (
            floor(
                (height + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1)
                / self.stride[0]
                + 1
            ),
            floor(
                (width + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1)
                / self.stride[1]
                + 1
            ),
        )

    def forward(self, x):
        return self.postarrange(self.unfold(self.prearrange(x)))

    def __repr__(self):
        return (
            f"Im2Col(kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding})"
        )


class Col2Im(nn.Module):
    """
    Rearrange the dimensions of a tensor to form an image.
    This is the inverse of the Im2Col class.
    It converts form the "col" mode to the "im" mode.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.rearrange = lambda x: x.permute(0, 2, 1).reshape(
            x.shape[0], -1, self.output_shape[0], self.output_shape[1]
        )

    def forward(self, x):
        x = self.rearrange(x)
        return x

    def __repr__(self):
        if hasattr(self, "output_shape"):
            return f"Col2Im(output_shape={self.output_shape})"
        else:
            return "Col2Im()"


class Permute(nn.Module):
    """Permute the dimensions of a tensor."""

    def __init__(self, dims, **kwargs):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(self.dims)

    def __repr__(self):
        return f"Permute(dims={self.dims})"


class DotProduct(nn.Module):
    """Dot product of two tensors with optional scaling."""

    def __init__(self, scaled=False, **kwargs):
        super().__init__()
        self.scaled = scaled

    def forward(self, tensors):
        a, b = tensors
        scale_factor = 1.0 / sqrt(a.size(-1)) if self.scaled else 1.0
        if a.dim() == 2 and b.dim() == 2:
            a, b = (a.unsqueeze(1), b.unsqueeze(-1))
            return (a @ b * scale_factor).squeeze(-1)
        else:
            return a @ b * scale_factor

    def __repr__(self):
        return f"DotProduct(scaled={self.scaled})"


class AddTensors(nn.Module):
    """Add tensors together."""

    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, tensors):
        out = torch.stack(tensors)
        out = out.sum(0)
        return out


class CatTensors(nn.Module):
    """Concatenate tensors along a specified dimension."""

    def __init__(self, dim, **kwargs):
        super().__init__()
        self.dim = dim

    def forward(self, tensors):
        out = torch.cat(tensors, dim=self.dim)
        return out

    def __repr__(self):
        return f"CatTensors(dim={self.dim})"


class BroadcastTensors(nn.Module):
    """
    Merge two tensors in the most general way to a common output shape.
    This is useful for broadcasting two tensors of different shapes into a single output.
    This works with any two tensors that share a common prefix in their shapes and have 3 or 4 dimensions.
    """

    def __init__(self, mode="add", **kwargs):
        super().__init__()
        self.mode = mode

    def forward(self, tensors):
        """
        Align and merge two tensors using the specified mode: 'add', 'cat', or 'matmul'.

        Parameters:
        - tensors: tuple of two tensors (a, b) to be merged.

        Returns:
        - Merged tensor based on the mode.
        """
        a, b = tensors
        assert a.dim() in [3, 4] and b.dim() in [3, 4], "Only 3D and 4D tensors are supported"
        while a.dim() < b.dim():
            a = a.unsqueeze(1)
        while b.dim() < a.dim():
            b = b.unsqueeze(1)
        a_dims, b_dims = (a.dim(), b.dim())
        for i in range(1, a_dims):
            for j in range(1, b_dims):
                if (
                    a.shape[i] == b.shape[j]
                    and a.shape[i] != b.shape[i]
                    and (a.shape[j] != b.shape[j])
                ):
                    dims = list(range(b.dim()))
                    dims[i], dims[j] = (dims[j], dims[i])
                    b = b.permute(dims)
        a_shape, b_shape = (list(a.shape), list(b.shape))
        for i in range(1, min(a_dims, b_dims) + 1):
            if a_shape[-i] == b_shape[-i]:
                continue
            elif a_shape[-i] == 1:
                a_shape[-i] = b_shape[-i]
                a = a.expand(a_shape)
            elif b_shape[-i] == 1:
                b_shape[-i] = a_shape[-i]
                b = b.expand(b_shape)
            else:
                b = b.mean(dim=-i, keepdim=True)
                b_shape[-i] = a_shape[-i]
                b = b.expand(b_shape)
        if self.mode == "add":
            return a + b
        elif self.mode == "cat":
            return torch.cat([a, b], dim=-1)
        elif self.mode == "matmul":
            b = b.permute(list(range(b.dim() - 2)) + [-1, -2])
            return torch.matmul(a, b)

    def __repr__(self):
        return f"BroadcastTensors(mode={self.mode})"


class EinLinear(nn.Module):
    def __init__(self, in_dim, out_dim, **kwargs):
        """Linear layer with input and output dimensions."""
        super().__init__()
        self.fn = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        out = self.fn(x)
        return out


class EinNorm(nn.Module):
    """LayerNorm for 3D and BatchNorm for 4D."""

    def __init__(self, input_shape, **kwargs):
        super().__init__()
        if len(input_shape) == 2:
            self.fn = nn.LayerNorm(input_shape[-1])
        elif len(input_shape) == 3:
            self.fn = nn.BatchNorm1d(input_shape[1])
        elif len(input_shape) == 4:
            self.fn = nn.BatchNorm2d(input_shape[1])
        else:
            raise NotImplementedError("Only shapes of (B, C, H, W) and (B, C, L) implemented.")

    def forward(self, x):
        return self.fn(x)


class LearnablePositionalEncoding(nn.Module):
    """Learnable Positional Encoding"""

    def __init__(self, input_shape, **kwargs):
        super().__init__()
        self.input_shape = input_shape
        if len(input_shape) == 3:
            self.fn = nn.Parameter(torch.randn(1, input_shape[1], input_shape[2]))
        elif len(input_shape) == 4:
            self.fn = nn.Parameter(torch.randn(1, input_shape[1], input_shape[2], input_shape[3]))
        else:
            raise NotImplementedError("Only shapes of (B, C, H, W) and (B, C, L) implemented.")

    def forward(self, x):
        return x + self.fn


class Sequential2(nn.Module):
    """
    A sequential module that takes in 2 inputs.
    """

    def __init__(self, first_fn, second_fn):
        super().__init__()
        self.first_fn = first_fn
        self.second_fn = second_fn

    def forward(self, x):
        out = self.first_fn(x)
        out = self.second_fn(out)
        return out


class Sequential3(nn.Module):
    """
    A sequential module that takes in 3 inputs.
    """

    def __init__(self, first_fn, second_fn, third_fn):
        super().__init__()
        self.first_fn = first_fn
        self.second_fn = second_fn
        self.third_fn = third_fn

    def forward(self, x):
        out = self.first_fn(x)
        out = self.second_fn(out)
        out = self.third_fn(out)
        return out


class Sequential4(nn.Module):
    """
    A sequential module that takes in 4 inputs.
    """

    def __init__(self, first_fn, second_fn, third_fn, fourth_fn):
        super().__init__()
        self.first_fn = first_fn
        self.second_fn = second_fn
        self.third_fn = third_fn
        self.fourth_fn = fourth_fn

    def forward(self, x):
        out = self.first_fn(x)
        out = self.second_fn(out)
        out = self.third_fn(out)
        out = self.fourth_fn(out)
        return out


class Residual2(nn.Module):
    """
    A residual module that takes in 2 inputs.
    """

    def __init__(self, first_fn, residual_fn, second_fn):
        super().__init__()
        self.first_fn = first_fn
        self.residual_fn = residual_fn
        self.second_fn = second_fn

    def forward(self, x):
        out = self.first_fn(x)
        out = self.second_fn(out)
        residual = self.residual_fn(x)
        out = out + residual
        return out


class Residual3(nn.Module):
    """
    A residual module that takes in 3 inputs.
    """

    def __init__(self, first_fn, second_fn, residual_fn, third_fn):
        super().__init__()
        self.first_fn = first_fn
        self.second_fn = second_fn
        self.residual_fn = residual_fn
        self.third_fn = third_fn

    def forward(self, x):
        out = self.first_fn(x)
        out = self.second_fn(out)
        out = self.third_fn(out)
        residual = self.residual_fn(x)
        out = out + residual
        return out


class Cell(nn.Module):
    """
    A NasBench201 cell module that has 6 input functions.
    """

    def __init__(self, a, b, c, d, e, f):
        super().__init__()
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.e = e
        self.f = f

    def forward(self, x):
        a_out = self.a(x)
        b_out = self.b(x)
        c_out = self.c(a_out)
        d_out = self.d(x)
        e_out = self.e(a_out)
        f_out = self.f(b_out + c_out)
        out = d_out + e_out + f_out
        return out


class Diamond2(nn.Module):
    """
    A diamond module that takes in 4 function inputs.
    """

    def __init__(self, a, b, c, d):
        super().__init__()
        self.a = a
        self.b = b
        self.c = c
        self.d = d

    def forward(self, x):
        a_out = self.a(x)
        b_out = self.b(x)
        c_out = self.c(a_out)
        d_out = self.d(b_out)
        out = c_out + d_out
        return out


class Diamond3(nn.Module):
    """
    A diamond module that takes in 6 function inputs.
    """

    def __init__(self, a, b, c, d, e, f):
        super().__init__()
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.e = e
        self.f = f

    def forward(self, x):
        a_out = self.a(x)
        b_out = self.b(x)
        c_out = self.c(a_out)
        d_out = self.d(b_out)
        e_out = self.e(c_out)
        f_out = self.f(d_out)
        out = e_out + f_out
        return out


class DownConv(nn.Module):
    """
    A downsampling convolutional layer.
    """

    def __init__(self, input_shape):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=input_shape[1],
            out_channels=input_shape[1] * 2,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.conv2 = nn.Conv2d(
            in_channels=input_shape[1] * 2,
            out_channels=input_shape[1] * 2,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.conv3 = nn.Conv2d(
            in_channels=input_shape[1],
            out_channels=input_shape[1] * 2,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(self, x):
        if x.shape[2] % 2 == 1:
            x = F.pad(x, (0, 0, 0, 1), mode="constant", value=0)
        if x.shape[3] % 2 == 1:
            x = F.pad(x, (0, 1, 0, 0), mode="constant", value=0)
        out1 = self.conv2(self.conv1(x))
        out2 = self.conv3(self.avgpool(x))
        return out1 + out2


# The hNAS grammar uses the original Lambda constructor.
nn.Lambda = Lambda
