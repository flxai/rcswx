"""Executable network pairs, separate from the exhaustive primitive inventory.

Size, parameter width, recursion and ambiguity are independent axes. Stress
recipes are not promises that exponentially large reference requests finish.
Every descriptor remains available when a measurement is interrupted.
"""

import copy
from dataclasses import asdict, dataclass

from .corpus import build_case, computation
from .original import load


@dataclass(frozen=True, slots=True)
class NetworkPair:
    name: str
    family: str
    category: str
    scale: int
    shape: tuple[int, ...]
    left: str | tuple
    right: str | tuple
    level: str = "network"

    def record(self):
        return asdict(self)


def sequence(modules):
    """Balanced source sequential productions, not an invented flat operator."""
    modules = tuple(modules)
    if not modules:
        raise ValueError("A network sequence needs at least one module")
    if len(modules) == 1:
        return modules[0]
    middle = len(modules) // 2
    return "sequential", sequence(modules[:middle]), sequence(modules[middle:])


def mlp(blocks, width=16, *, normalized=False, changed=False):
    modules = []
    for index in range(blocks):
        modules.append(computation(f"linear({width})"))
        modules.append(computation("softmax" if changed and index == blocks // 2 else "relu"))
        if normalized:
            modules.append(computation("norm"))
    return sequence(modules)


def relative_mlp(blocks, *, changed=False):
    modules = []
    for index in range(blocks):
        modules.extend(
            (computation("linear(x2)"), computation("relu"), computation("linear(x0.5)"))
        )
        modules.append(computation("norm" if changed and index == blocks // 2 else "identity"))
    return sequence(modules)


def convolution(width=16, *, activation="relu"):
    return sequence(
        (
            ("routing", "im2col(3,1,1)", computation(f"linear({width})"), "col2im"),
            computation("norm"),
            computation(activation),
        )
    )


def residual(blocks, *, changed=False):
    return sequence(
        (
            "branching(2)",
            "clone(2)",
            mlp(1, changed=changed and index == blocks // 2),
            computation("identity"),
            "add(2)",
        )
        for index in range(blocks)
    )


def grouped(arity, *, changed=False):
    inner = sequence((computation("norm"), computation("softmax" if changed else "relu")))
    return (
        f"branching({arity})",
        f"group({arity},2)",
        inner,
        *([inner] if arity == 2 else []),
        f"cat({arity},2)",
    )


def attention(*, changed=False):
    query = mlp(1)
    key = ("routing", "identity", mlp(1), "perm(0,2,1)")
    scores = ("branching(2)", "clone(2)", query, key, "dot_product(scaled)")
    return sequence(
        (scores, computation("softmax"), computation("linear(32)" if changed else "linear(16)"))
    )


def nested_routing(depth, *, changed=False):
    inner = mlp(1, changed=changed)
    for _ in range(depth):
        inner = ("routing", "perm(0,2,1)", inner, "perm(0,2,1)")
    return inner


def hnas_network(*, dense=False, changed=False):
    grammar = load().hnasbench201.grammar
    defaults = {level: rules["options"][0] for level, rules in grammar.items()}

    def expand(operation):
        if operation.name == "cell_OP_OP_OP_OP_OP_OP":
            block = (
                "sequential3_ACT_CONV_NORM",
                "relu",
                "dconv3x3" if changed else "conv3x3",
                "batchnorm",
            )
            edges = [block if dense or edge == 0 else "identity" for edge in range(6)]
            return operation.name, *edges
        if not operation.child_levels:
            return operation.name
        return operation.name, *(expand(defaults[level]) for level in operation.child_levels)

    return expand(defaults["network"])


def network_pairs(suite="smoke"):
    if suite not in {"smoke", "scaling", "all"}:
        raise ValueError(f"Unknown network suite: {suite}")
    cases = {}

    def add(
        name, category, scale, left, right, *, shape=(2, 8, 16), family="einspace", level="network"
    ):
        cases[name] = NetworkPair(name, family, category, scale, shape, left, right, level)

    if suite in {"smoke", "all"}:
        add("mlp-small", "mlp_depth", 2, mlp(2), mlp(2, changed=True))
        add("mlp-depth-crossover", "unequal_mlp_depth", 3, mlp(2), mlp(3))
        add(
            "normalized-depth-crossover",
            "unequal_normalized_depth",
            2,
            mlp(1, normalized=True),
            mlp(2, normalized=True),
        )
        add(
            "normalized-mlp",
            "normalization",
            2,
            mlp(2, normalized=True),
            mlp(2, normalized=True, changed=True),
        )
        add(
            "relative-width-mlp",
            "relative_width",
            2,
            relative_mlp(2),
            relative_mlp(2, changed=True),
        )
        add(
            "unfold-cnn",
            "image_routing",
            1,
            convolution(),
            convolution(activation="softmax"),
            shape=(2, 8, 8, 8),
        )
        add(
            "cnn-depth-crossover",
            "unequal_cnn_depth",
            2,
            convolution(),
            sequence([convolution()] * 2),
            shape=(2, 8, 8, 8),
        )
        add("residual-block", "binary_residual", 1, residual(1), residual(1, changed=True))
        for arity in (4, 8):
            add(
                f"repeated-branches-{arity}",
                "branch_arity",
                arity,
                (f"branching({arity})", f"clone({arity})", mlp(1), f"add({arity})"),
                (f"branching({arity})", f"clone({arity})", mlp(1, changed=True), f"add({arity})"),
            )
            add(
                f"grouped-concat-{arity}",
                "grouped_concat",
                arity,
                grouped(arity),
                grouped(arity, changed=True),
            )
        add(
            "attention-token-mixing",
            "attention",
            1,
            attention(),
            attention(changed=True),
            shape=(2, 8, 16),
        )
        add(
            "nested-routing",
            "routing_depth",
            2,
            nested_routing(2),
            nested_routing(2, changed=True),
            shape=(2, 16, 16),
        )
        add(
            "repetition-macro",
            "macro_repetition",
            4,
            ("sequential(4)", mlp(1)),
            ("sequential(4)", mlp(1, changed=True)),
        )
        add(
            "macro-versus-explicit",
            "macro_identity",
            4,
            ("sequential(4)", mlp(1)),
            sequence([mlp(1)] * 4),
        )
        for dense in (False, True):
            add(
                f"hnas-{'dense' if dense else 'sparse'}-cnn",
                "hnas_convolution",
                6 if dense else 1,
                hnas_network(dense=dense),
                hnas_network(dense=dense, changed=True),
                shape=(2, 4, 16, 16),
                family="hnasbench201",
            )
        add(
            "ties-small",
            "tie_density",
            4,
            sequence([computation()] * 4),
            sequence([computation()] * 2),
        )
        add(
            "unique-history-small",
            "unique_history",
            8,
            sequence([computation("relu")] + [computation()] * 6 + [computation("softmax")]),
            sequence([computation("relu"), computation("softmax")]),
        )

    if suite in {"scaling", "all"}:
        for size in (2, 4, 8, 16, 32):
            add(f"mlp-depth-{size}", "mlp_depth", size, mlp(size), mlp(size, changed=True))
            add(
                f"normalized-depth-{size}",
                "normalized_depth",
                size,
                mlp(size, normalized=True),
                mlp(size, normalized=True, changed=True),
            )
            add(
                f"relative-depth-{size}",
                "relative_width",
                size,
                relative_mlp(size),
                relative_mlp(size, changed=True),
            )
        for width in (16, 64, 256, 512):
            add(
                f"parameter-width-{width}",
                "parameter_width",
                width,
                mlp(4, width, normalized=True),
                mlp(4, width, normalized=True, changed=True),
                shape=(2, 8, width),
            )
        for depth in (1, 2, 4, 8):
            add(
                f"routing-depth-{depth}",
                "routing_depth",
                depth,
                nested_routing(depth),
                nested_routing(depth, changed=True),
                shape=(2, 16, 16),
            )
            add(
                f"residual-depth-{depth}",
                "binary_residual",
                depth,
                residual(depth),
                residual(depth, changed=True),
            )
            add(
                f"cnn-depth-{depth}",
                "image_routing",
                depth,
                sequence([convolution()] * depth),
                sequence([convolution()] * (depth - 1) + [convolution(activation="softmax")]),
                shape=(2, 8, 8, 8),
            )
        for arity in (2, 4, 8):
            left = (
                f"branching({arity})",
                f"clone({arity})",
                mlp(1),
                *([mlp(1)] if arity == 2 else []),
                f"add({arity})",
            )
            right = (
                f"branching({arity})",
                f"clone({arity})",
                mlp(1, changed=True),
                *([mlp(1)] if arity == 2 else []),
                f"add({arity})",
            )
            add(f"branch-arity-{arity}", "branch_arity", arity, left, right)
        for repeat in (4, 8):
            add(
                f"macro-repeat-{repeat}",
                "macro_repetition",
                repeat,
                (f"sequential({repeat})", mlp(1)),
                (f"sequential({repeat})", mlp(1, changed=True)),
            )
        for size in (4, 8, 16, 32, 64, 128):
            add(
                f"unique-history-{size}",
                "unique_history",
                size,
                sequence(
                    [computation("relu")] + [computation()] * (size - 2) + [computation("softmax")]
                ),
                sequence([computation("relu"), computation("softmax")]),
            )
        for size in (4, 8, 12, 16, 20, 24, 32):
            add(
                f"dense-ties-{size}",
                "tie_size",
                size,
                sequence([computation()] * size),
                sequence([computation()] * (size // 2)),
            )
        for retained in (16, 12, 8, 4, 2):
            add(
                f"tie-density-16-{retained}",
                "tie_density",
                retained,
                sequence([computation()] * 16),
                sequence([computation()] * retained),
            )
    return cases


def prepare_pair(case, owned):
    """Sample actual genotypes without building/warming the target models."""
    parents = tuple(
        build_case(
            (case.family, description, case.shape, case.level), owned=owned, build_model=False
        )[0]
        for description in (case.left, case.right)
    )
    guard = parents[0].limiter
    reference = load()
    if owned:
        from rcswx import PCFG, Reconstructor
        from rcswx.grammars import einspace, hnasbench201

        module = {"einspace": einspace, "hnasbench201": hnasbench201}[case.family]
        pcfg_type, builder_type = PCFG, Reconstructor
    else:
        module = getattr(reference, case.family)
        pcfg_type, builder_type = reference.pcfg.PCFG, reference.evolution.Evolver
    grammar = copy.deepcopy(
        module.deep_broadcast_grammar if case.family == "einspace" else module.grammar
    )
    if case.family == "einspace":
        for factor in (0.5, 2):
            grammar["computation_fn"]["options"].append(module.linear_x(factor))
            grammar["computation_fn"]["probs"].append(0.077)
    builder = builder_type(pcfg=pcfg_type(grammar, guard), limiter=guard, mode="iterative")
    return parents, builder, guard
