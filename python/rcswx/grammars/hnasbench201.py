# Derived from einsearch 7e713c7951397a6b12bd57638409774a74381746.
# Copyright (c) 2024 Adri Gómez Martín, Felix Möller, Linus Ericsson, Aaron Klein.
# Distributed under the MIT license; see LICENSE.einsearch.
from functools import partial

import torch
import torch.nn as nn

from .. import layers
from ..genotype import Operation


def inherit_first_child(node):
    node.input_params = node.parent.input_params


def inherit_other_child(node):
    node.input_params = node.parent.output_params


def inherit_ith_child(i, node):
    node.input_params = node.parent.children[i].output_params


def inherit_branching(node):
    node.input_params = node.parent.children[0].output_params


def inherit_aggregation(node):
    if len(node.parent.children) >= 4:
        node.input_params = node.parent.children[1].output_params
        node.input_params["other_shape"] = node.parent.children[2].output_params["shape"]
        node.input_params["other_mode"] = node.parent.children[2].output_params["mode"]
    elif len(node.parent.children) >= 3:
        node.input_params = node.parent.children[1].output_params
        node.input_params["other_shape"] = node.parent.children[1].output_params["shape"]
        node.input_params["other_mode"] = node.parent.children[1].output_params["mode"]


def give_back_default(node):
    node.parent.output_params = node.output_params


def build_sequential4(node):
    return layers.Sequential4(
        first_fn=node.children[0].build(node.children[0]),
        second_fn=node.children[1].build(node.children[1]),
        third_fn=node.children[2].build(node.children[2]),
        fourth_fn=node.children[3].build(node.children[3]),
    )


def infer_sequential(node):
    return node.input_params


def valid_sequential(node):
    return True


sequential4_D1_D1_D0_D0 = Operation(
    name="sequential4_D1_D1_D0_D0",
    build=build_sequential4,
    infer=infer_sequential,
    valid=valid_sequential,
    inherit=[inherit_first_child, inherit_other_child, inherit_other_child, inherit_other_child],
    give_back=[give_back_default, give_back_default, give_back_default, give_back_default],
    type="nonterminal",
    child_levels=["D1", "D1", "D0", "D0"],
)


def build_sequential3(node):
    return layers.Sequential3(
        first_fn=node.children[0].build(node.children[0]),
        second_fn=node.children[1].build(node.children[1]),
        third_fn=node.children[2].build(node.children[2]),
    )


sequential3_D1_D1_D0 = Operation(
    name="sequential3_D1_D1_D0",
    build=build_sequential3,
    infer=infer_sequential,
    valid=valid_sequential,
    inherit=[inherit_first_child, inherit_other_child, inherit_other_child],
    give_back=[give_back_default, give_back_default, give_back_default],
    type="nonterminal",
    child_levels=["D1", "D1", "D0"],
)
sequential3_D0_D1_D1 = Operation(
    name="sequential3_D0_D1_D1",
    build=build_sequential3,
    infer=infer_sequential,
    valid=valid_sequential,
    inherit=[inherit_first_child, inherit_other_child, inherit_other_child],
    give_back=[give_back_default, give_back_default, give_back_default],
    type="nonterminal",
    child_levels=["D0", "D1", "D1"],
)
sequential3_C_C_D = Operation(
    name="sequential3_C_C_D",
    build=build_sequential3,
    infer=infer_sequential,
    valid=valid_sequential,
    inherit=[inherit_first_child, inherit_other_child, inherit_other_child],
    give_back=[give_back_default, give_back_default, give_back_default],
    type="nonterminal",
    child_levels=["C", "C", "D"],
)
sequential3_C_C_CL = Operation(
    name="sequential3_C_C_CL",
    build=build_sequential3,
    infer=infer_sequential,
    valid=valid_sequential,
    inherit=[inherit_first_child, inherit_other_child, inherit_other_child],
    give_back=[give_back_default, give_back_default, give_back_default],
    type="nonterminal",
    child_levels=["C", "C", "CL"],
)


def build_sequential2(node):
    return layers.Sequential2(
        first_fn=node.children[0].build(node.children[0]),
        second_fn=node.children[1].build(node.children[1]),
    )


sequential2_CL_DOWN = Operation(
    name="sequential2_CL_DOWN",
    build=build_sequential2,
    infer=infer_sequential,
    valid=valid_sequential,
    inherit=[inherit_first_child, inherit_other_child],
    give_back=[give_back_default, give_back_default],
    type="nonterminal",
    child_levels=["CL", "DOWN"],
)
sequential2_CL_CL = Operation(
    name="sequential2_CL_CL",
    build=build_sequential2,
    infer=infer_sequential,
    valid=valid_sequential,
    inherit=[inherit_first_child, inherit_other_child],
    give_back=[give_back_default, give_back_default],
    type="nonterminal",
    child_levels=["CL", "CL"],
)
sequential3_ACT_CONV_NORM = Operation(
    name="sequential3_ACT_CONV_NORM",
    build=build_sequential3,
    infer=infer_sequential,
    valid=valid_sequential,
    inherit=[inherit_first_child, inherit_other_child, inherit_other_child],
    give_back=[give_back_default, give_back_default, give_back_default],
    type="nonterminal",
    child_levels=["ACT", "CONV", "NORM"],
)


def build_residual3(node):
    return layers.Residual3(
        first_fn=node.children[0].build(node.children[0]),
        second_fn=node.children[1].build(node.children[1]),
        residual_fn=node.children[2].build(node.children[2]),
        third_fn=node.children[3].build(node.children[3]),
    )


def infer_residual(node):
    return node.input_params


def valid_residual(node):
    return True


residual3_C_C_D_D = Operation(
    name="residual3_C_C_D_D",
    build=build_residual3,
    infer=infer_residual,
    valid=valid_residual,
    inherit=[
        inherit_first_child,
        partial(inherit_ith_child, 0),
        inherit_first_child,
        partial(inherit_ith_child, 1),
    ],
    give_back=[give_back_default, give_back_default, give_back_default, give_back_default],
    type="nonterminal",
    child_levels=["C", "C", "D", "D"],
)
residual3_C_C_CL_CL = Operation(
    name="residual3_C_C_CL_CL",
    build=build_residual3,
    infer=infer_residual,
    valid=valid_residual,
    inherit=[
        inherit_first_child,
        partial(inherit_ith_child, 0),
        inherit_first_child,
        partial(inherit_ith_child, 1),
    ],
    give_back=[give_back_default, give_back_default, give_back_default, give_back_default],
    type="nonterminal",
    child_levels=["C", "C", "CL", "CL"],
)


def build_residual2(node):
    return layers.Residual2(
        first_fn=node.children[0].build(node.children[0]),
        residual_fn=node.children[1].build(node.children[1]),
        second_fn=node.children[2].build(node.children[2]),
    )


residual2_CL_DOWN_DOWN = Operation(
    name="residual2_CL_DOWN_DOWN",
    build=build_residual2,
    infer=infer_residual,
    valid=valid_residual,
    inherit=[inherit_first_child, inherit_first_child, partial(inherit_ith_child, 0)],
    give_back=[give_back_default, give_back_default, give_back_default],
    type="nonterminal",
    child_levels=["CL", "DOWN", "DOWN"],
)
residual2_CL_CL_CL = Operation(
    name="residual2_CL_CL_CL",
    build=build_residual2,
    infer=infer_residual,
    valid=valid_residual,
    inherit=[inherit_first_child, inherit_first_child, partial(inherit_ith_child, 0)],
    give_back=[give_back_default, give_back_default, give_back_default],
    type="nonterminal",
    child_levels=["CL", "CL", "CL"],
)


def build_diamond3(node):
    return layers.Diamond3(
        a=node.children[0].build(node.children[0]),
        b=node.children[1].build(node.children[1]),
        c=node.children[2].build(node.children[2]),
        d=node.children[3].build(node.children[3]),
        e=node.children[4].build(node.children[4]),
        f=node.children[5].build(node.children[5]),
    )


def infer_diamond(node):
    return node.input_params


def valid_diamond(node):
    return True


diamond3_C_C_C_C_D_D = Operation(
    name="diamond3_C_C_C_C_D_D",
    build=build_diamond3,
    infer=infer_diamond,
    valid=valid_diamond,
    inherit=[
        inherit_first_child,
        inherit_first_child,
        partial(inherit_ith_child, 0),
        partial(inherit_ith_child, 1),
        partial(inherit_ith_child, 2),
        partial(inherit_ith_child, 3),
    ],
    give_back=[
        give_back_default,
        give_back_default,
        give_back_default,
        give_back_default,
        give_back_default,
        give_back_default,
    ],
    type="nonterminal",
    child_levels=["C", "C", "C", "C", "D", "D"],
)
diamond3_C_C_C_C_CL_CL = Operation(
    name="diamond3_C_C_C_C_CL_CL",
    build=build_diamond3,
    infer=infer_diamond,
    valid=valid_diamond,
    inherit=[
        inherit_first_child,
        inherit_first_child,
        partial(inherit_ith_child, 0),
        partial(inherit_ith_child, 1),
        partial(inherit_ith_child, 2),
        partial(inherit_ith_child, 3),
    ],
    give_back=[
        give_back_default,
        give_back_default,
        give_back_default,
        give_back_default,
        give_back_default,
        give_back_default,
    ],
    type="nonterminal",
    child_levels=["C", "C", "C", "C", "CL", "CL"],
)


def build_diamond2(node):
    return layers.Diamond2(
        a=node.children[0].build(node.children[0]),
        b=node.children[1].build(node.children[1]),
        c=node.children[2].build(node.children[2]),
        d=node.children[3].build(node.children[3]),
    )


diamond2_CL_CL_DOWN_DOWN = Operation(
    name="diamond2_CL_CL_DOWN_DOWN",
    build=build_diamond2,
    infer=infer_diamond,
    valid=valid_diamond,
    inherit=[
        inherit_first_child,
        inherit_first_child,
        partial(inherit_ith_child, 0),
        partial(inherit_ith_child, 1),
    ],
    give_back=[give_back_default, give_back_default, give_back_default, give_back_default],
    type="nonterminal",
    child_levels=["CL", "CL", "DOWN", "DOWN"],
)
diamond2_CL_CL_CL_CL = Operation(
    name="diamond2_CL_CL_CL_CL",
    build=build_diamond2,
    infer=infer_diamond,
    valid=valid_diamond,
    inherit=[
        inherit_first_child,
        inherit_first_child,
        partial(inherit_ith_child, 0),
        partial(inherit_ith_child, 1),
    ],
    give_back=[give_back_default, give_back_default, give_back_default, give_back_default],
    type="nonterminal",
    child_levels=["CL", "CL", "CL", "CL"],
)


def build_cell(node):
    return layers.Cell(
        a=node.children[0].build(node.children[0]),
        b=node.children[1].build(node.children[1]),
        c=node.children[2].build(node.children[2]),
        d=node.children[3].build(node.children[3]),
        e=node.children[4].build(node.children[4]),
        f=node.children[5].build(node.children[5]),
    )


def infer_cell(node):
    return node.input_params


def valid_cell(node):
    return True


cell_OP_OP_OP_OP_OP_OP = Operation(
    name="cell_OP_OP_OP_OP_OP_OP",
    build=build_cell,
    infer=infer_cell,
    valid=valid_cell,
    inherit=[
        inherit_first_child,
        inherit_first_child,
        partial(inherit_ith_child, 0),
        inherit_first_child,
        partial(inherit_ith_child, 0),
        partial(inherit_ith_child, 1),
    ],
    give_back=[
        give_back_default,
        give_back_default,
        give_back_default,
        give_back_default,
        give_back_default,
        give_back_default,
    ],
    type="nonterminal",
    child_levels=["OP", "OP", "OP", "OP", "OP", "OP"],
)


def build_down(node):
    return layers.DownConv(input_shape=node.input_params["shape"])


def infer_down(node):
    return {
        "shape": (
            node.input_params["shape"][0],
            node.input_params["shape"][1] * 2,
            node.input_params["shape"][2] // 2,
            node.input_params["shape"][3] // 2,
        ),
        "other_shape": node.input_params["other_shape"],
        "mode": node.input_params["mode"],
        "other_mode": node.input_params["other_mode"],
        "branching_factor": node.input_params["branching_factor"],
        "last_im_shape": node.input_params["last_im_shape"],
    }


def valid_down(node):
    return node.input_params["shape"][2] > 1 and node.input_params["shape"][3] > 1


down = Operation(
    name="down",
    build=build_down,
    infer=infer_down,
    valid=valid_down,
    inherit=[inherit_first_child],
    give_back=[give_back_default],
    type="terminal",
    child_levels=[],
)


def build_zero(node):
    return nn.Lambda(lambda x: torch.zeros_like(x).to(x.device))


def infer_zero(node):
    return node.input_params


def valid_zero(node):
    return True


zero = Operation(
    name="zero",
    build=build_zero,
    infer=infer_zero,
    valid=valid_zero,
    inherit=[inherit_first_child],
    give_back=[give_back_default],
    type="terminal",
    child_levels=[],
)


def build_identity(node):
    return nn.Identity()


def infer_identity(node):
    return node.input_params


def valid_identity(node):
    return True


identity = Operation(
    name="identity",
    build=build_identity,
    infer=infer_identity,
    valid=valid_identity,
    inherit=[inherit_first_child],
    give_back=[give_back_default],
    type="terminal",
    child_levels=[],
)


def build_avg_pool(node):
    return nn.AvgPool2d(kernel_size=3, stride=1, padding=1)


def infer_avg_pool(node):
    return node.input_params


def valid_avg_pool(node):
    return True


avg_pool = Operation(
    name="avg_pool",
    build=build_avg_pool,
    infer=infer_avg_pool,
    valid=valid_avg_pool,
    inherit=[inherit_first_child],
    give_back=[give_back_default],
    type="terminal",
    child_levels=[],
)


def build_relu(node):
    return nn.ReLU()


def infer_relu(node):
    return node.input_params


def valid_relu(node):
    return True


relu = Operation(
    name="relu",
    build=build_relu,
    infer=infer_relu,
    valid=valid_relu,
    inherit=[inherit_first_child],
    give_back=[give_back_default],
    type="terminal",
    child_levels=[],
)


def build_hardswish(node):
    return nn.Hardswish()


def infer_hardswish(node):
    return node.input_params


def valid_hardswish(node):
    return True


hardswish = Operation(
    name="hardswish",
    build=build_hardswish,
    infer=infer_hardswish,
    valid=valid_hardswish,
    inherit=[inherit_first_child],
    give_back=[give_back_default],
    type="terminal",
    child_levels=[],
)


def build_mish(node):
    return nn.Hardswish()


def infer_mish(node):
    return node.input_params


def valid_mish(node):
    return True


mish = Operation(
    name="mish",
    build=build_mish,
    infer=infer_mish,
    valid=valid_mish,
    inherit=[inherit_first_child],
    give_back=[give_back_default],
    type="terminal",
    child_levels=[],
)


def build_conv1x1(node):
    return nn.Conv2d(
        in_channels=node.input_params["shape"][1],
        out_channels=node.input_params["shape"][1],
        kernel_size=1,
        stride=1,
        padding=0,
    )


def infer_conv1x1(node):
    return node.input_params


def valid_conv1x1(node):
    return True


conv1x1 = Operation(
    name="conv1x1",
    build=build_conv1x1,
    infer=infer_conv1x1,
    valid=valid_conv1x1,
    inherit=[inherit_first_child],
    give_back=[give_back_default],
    type="terminal",
    child_levels=[],
)


def build_conv3x3(node):
    return nn.Conv2d(
        in_channels=node.input_params["shape"][1],
        out_channels=node.input_params["shape"][1],
        kernel_size=3,
        stride=1,
        padding=1,
    )


def infer_conv3x3(node):
    return node.input_params


def valid_conv3x3(node):
    return True


conv3x3 = Operation(
    name="conv3x3",
    build=build_conv3x3,
    infer=infer_conv3x3,
    valid=valid_conv3x3,
    inherit=[inherit_first_child],
    give_back=[give_back_default],
    type="terminal",
    child_levels=[],
)


def build_dconv3x3(node):
    return nn.Conv2d(
        in_channels=node.input_params["shape"][1],
        out_channels=node.input_params["shape"][1],
        kernel_size=3,
        stride=1,
        padding=1,
        groups=node.input_params["shape"][1],
    )


def infer_dconv3x3(node):
    return node.input_params


def valid_dconv3x3(node):
    return True


dconv3x3 = Operation(
    name="dconv3x3",
    build=build_dconv3x3,
    infer=infer_dconv3x3,
    valid=valid_dconv3x3,
    inherit=[inherit_first_child],
    give_back=[give_back_default],
    type="terminal",
    child_levels=[],
)


def build_batchnorm(node):
    return nn.BatchNorm2d(node.input_params["shape"][1])


def infer_batchnorm(node):
    return node.input_params


def valid_batchnorm(node):
    return True


batchnorm = Operation(
    name="batchnorm",
    build=build_batchnorm,
    infer=infer_batchnorm,
    valid=valid_batchnorm,
    inherit=[inherit_first_child],
    give_back=[give_back_default],
    type="terminal",
    child_levels=[],
)


def build_instancenorm(node):
    return nn.InstanceNorm2d(node.input_params["shape"][1])


def infer_instancenorm(node):
    return node.input_params


def valid_instancenorm(node):
    return True


instancenorm = Operation(
    name="instancenorm",
    build=build_instancenorm,
    infer=infer_instancenorm,
    valid=valid_instancenorm,
    inherit=[inherit_first_child],
    give_back=[give_back_default],
    type="terminal",
    child_levels=[],
)


def build_layernorm(node):
    return nn.LayerNorm(node.input_params["shape"][1:])


def infer_layernorm(node):
    return node.input_params


def valid_layernorm(node):
    return True


layernorm = Operation(
    name="layernorm",
    build=build_layernorm,
    infer=infer_layernorm,
    valid=valid_layernorm,
    inherit=[inherit_first_child],
    give_back=[give_back_default],
    type="terminal",
    child_levels=[],
)
D2_options = {
    "options": [sequential3_D1_D1_D0, sequential3_D0_D1_D1, sequential4_D1_D1_D0_D0],
    "probs": [1 / 3, 1 / 3, 1 / 3],
}
D1_options = {
    "options": [sequential3_C_C_D, residual3_C_C_D_D, diamond3_C_C_C_C_D_D],
    "probs": [1 / 3, 1 / 3, 1 / 3],
}
D0_options = {
    "options": [sequential3_C_C_CL, residual3_C_C_CL_CL, diamond3_C_C_C_C_CL_CL],
    "probs": [1 / 3, 1 / 3, 1 / 3],
}
D_options = {
    "options": [sequential2_CL_DOWN, residual2_CL_DOWN_DOWN, diamond2_CL_CL_DOWN_DOWN],
    "probs": [1 / 3, 1 / 3, 1 / 3],
}
C_options = {
    "options": [sequential2_CL_CL, residual2_CL_CL_CL, diamond2_CL_CL_CL_CL],
    "probs": [1 / 3, 1 / 3, 1 / 3],
}
CL_options = {"options": [cell_OP_OP_OP_OP_OP_OP], "probs": [1]}
OP_options = {
    "options": [zero, identity, sequential3_ACT_CONV_NORM, avg_pool],
    "probs": [1 / 4, 1 / 4, 1 / 4, 1 / 4],
}
ACT_options = {"options": [relu, hardswish, mish], "probs": [1 / 3, 1 / 3, 1 / 3]}
CONV_options = {"options": [conv1x1, conv3x3, dconv3x3], "probs": [1 / 3, 1 / 3, 1 / 3]}
NORM_options = {"options": [batchnorm, instancenorm, layernorm], "probs": [1 / 3, 1 / 3, 1 / 3]}
DOWN_options = {"options": [down], "probs": [1]}
grammar = {
    "network": D2_options,
    "D1": D1_options,
    "D0": D0_options,
    "D": D_options,
    "C": C_options,
    "CL": CL_options,
    "OP": OP_options,
    "ACT": ACT_options,
    "CONV": CONV_options,
    "NORM": NORM_options,
    "DOWN": DOWN_options,
}
