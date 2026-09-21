"""Pinned, unmodified source definitions used only as an independent test oracle.

Imports unrelated to RCSWX (datasets, plotting, trainers and MCTS diagnostics) are
not executed. Required classes/functions and complete grammar registries execute
unchanged, with their real NumPy/SciPy/Torch dependencies. This is not a claim to
reproduce a full NAS training environment or CUDA behavior on a CPU machine.
"""

import ast
import copy
import ctypes
import gc
import hashlib
import importlib.metadata
import math
import os
import pickle
import random
import sys
import time
import types
from collections import deque
from functools import lru_cache, reduce
from pathlib import Path

import numpy as np
import psutil
import rich
import torch
from scipy.stats import skewnorm
from termcolor import colored
from tqdm import tqdm

SOURCE_COMMIT = "7e713c7951397a6b12bd57638409774a74381746"
SOURCE_HASHES = {
    "search_state.py": "85d33ad71173c14dad3dedbfd7f847479b52fb4121e3e72ef4f09eb2a2a89ae1",
    "pcfg.py": "015f2a939c156c55a0228734eb41205bdb9412bda6106c27551890137156e988",
    "utils.py": "c9afc742fa74f448d2a7bdafb66aac7ea472ed4eb426c4660048a837744981fb",
    "layers.py": "df3b69dbf0a5725578fb723d3aed196b92c5438d9a4c299fa5ea605b8464ec71",
    "grammars/einspace.py": "de96676e01c15197bfb39df481129a062141bb1789e0a5fa7cb5fb08977a929b",
    "grammars/hnasbench201.py": "39cd185faaf14dc70ff233d17cbb3b45a66507d514d90cd5f197bc847b166a30",
    "search_strategies/random_search.py": "22ececd9a055f69de7c2645364c788544510c74bfd2b3bbbcc2e50f3b307586e",
    "search_strategies/evolution.py": "85e147e94fc9bfb996bee7308635ae1395505dcd0122ea50c78b25c4907747ba",
    "search_strategies/utils/recursive_constrained_smith_waterman.py": "f771b61a1bb5e182d5322138449f9886ed0999a1afb0c256c362b04887ef2876",
}


def environment():
    return {
        "python": sys.version,
        "numpy": np.__version__,
        "scipy": importlib.metadata.version("scipy"),
        "torch": torch.__version__,
        "device": "cpu",
        "default_dtype": str(torch.get_default_dtype()),
        "threads": torch.get_num_threads(),
        "source_commit": SOURCE_COMMIT,
        "source_hashes": SOURCE_HASHES,
    }


@lru_cache(maxsize=1)
def load():
    root = Path(
        os.environ.get("RCSWX_REFERENCE_ROOT", Path(__file__).resolve().parents[3] / "einsearch")
    )
    sources = {}
    for relative, expected in SOURCE_HASHES.items():
        path = root / relative
        contents = path.read_bytes()
        if hashlib.sha256(contents).hexdigest() != expected:
            raise RuntimeError(f"Reference source differs from pinned input: {path}")
        sources[relative] = ast.parse(contents, filename=str(path))

    package = types.ModuleType("_rcswx_original")
    package.__path__ = []
    sys.modules[package.__name__] = package

    def module(name, relative, namespace, names=None, exclude_imports=()):
        value = types.ModuleType(f"_rcswx_original.{name}")
        value.__dict__.update(namespace)
        sys.modules[value.__name__] = value
        if names is not None:
            body = [
                node
                for node in sources[relative].body
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names
            ]
        else:
            body = [
                node
                for node in sources[relative].body
                if not (
                    isinstance(node, ast.ImportFrom)
                    and node.module in exclude_imports
                    or isinstance(node, ast.Import)
                    and any(alias.name in exclude_imports for alias in node.names)
                )
            ]
        exec(
            compile(ast.Module(body=body, type_ignores=[]), str(root / relative), "exec"),
            value.__dict__,
        )
        return value

    pcfg = module(
        "pcfg",
        "pcfg.py",
        {"deepcopy": copy.deepcopy, "choices": random.choices, "sys": sys, "print": rich.print},
        {"PCFG", "OutOfOptionsError"},
    )
    state = module(
        "state",
        "search_state.py",
        {
            "deepcopy": copy.deepcopy,
            "sys": sys,
            "time": time,
            "psutil": psutil,
            "tqdm": tqdm,
            "print": rich.print,
            "OutOfOptionsError": pcfg.OutOfOptionsError,
        },
        {"Operation", "DerivationTreeNode", "Stack"},
    )
    utility = module(
        "utility",
        "utils.py",
        {
            "time": time.perf_counter,
            "psutil": psutil,
            "torch": torch,
            "ctypes": ctypes,
            "reduce": reduce,
            "gc": gc,
            "sys": sys,
        },
        {"Timer", "Limiter"},
    )
    layers = module(
        "layers",
        "layers.py",
        {
            "torch": torch,
            "nn": torch.nn,
            "F": torch.nn.functional,
            "floor": math.floor,
            "sqrt": math.sqrt,
            "log": math.log,
            "fftn": torch.fft.fftn,
            "ifftn": torch.fft.ifftn,
        },
        {
            node.name
            for node in sources["layers.py"].body
            if isinstance(node, (ast.ClassDef, ast.FunctionDef))
        },
    )
    torch.nn.Lambda = layers.Lambda
    einspace = module(
        "einspace",
        "grammars/einspace.py",
        {"Operation": state.Operation, "layers": layers},
        exclude_imports={"search_state", "layers"},
    )
    hnasbench201 = module(
        "hnasbench201",
        "grammars/hnasbench201.py",
        {"Operation": state.Operation, "layers": layers},
        exclude_imports={"search_state", "layers"},
    )
    sampler = module(
        "sampler",
        "search_strategies/random_search.py",
        {
            "DerivationTreeNode": state.DerivationTreeNode,
            "Stack": state.Stack,
            "OutOfOptionsError": pcfg.OutOfOptionsError,
        },
        {"Sampler"},
    )
    algorithm = module(
        "algorithm",
        "search_strategies/utils/recursive_constrained_smith_waterman.py",
        {
            "np": np,
            "copy": copy,
            "gc": gc,
            "psutil": psutil,
            "random": random,
            "time": time,
            "skewnorm": skewnorm,
            "colored": colored,
            "Operation": state.Operation,
            "DerivationTreeNode": state.DerivationTreeNode,
            "Limiter": utility.Limiter,
            "einspace": einspace,
        },
        {
            "MatrixCell",
            "MatrixOperation",
            "DecoyOperation",
            "DecoyNode",
            "AlignmentMatrixRecursive",
            "select_operations",
            "recursive_constrained_smith_waterman_crossover",
            "rcswx_distance",
        },
    )
    selector = copy.deepcopy(
        next(
            node
            for node in sources[
                "search_strategies/utils/recursive_constrained_smith_waterman.py"
            ].body
            if isinstance(node, ast.FunctionDef) and node.name == "select_operations"
        )
    )
    selector.name = "combinations"
    selector.body = selector.body[:2] + [
        ast.Return(value=ast.Name(id="combinations", ctx=ast.Load()))
    ]
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[selector], type_ignores=[])),
            str(root / "search_strategies/utils/recursive_constrained_smith_waterman.py"),
            "exec",
        ),
        algorithm.__dict__,
    )
    evolution = module(
        "evolution",
        "search_strategies/evolution.py",
        {
            "deque": deque,
            "Sampler": sampler.Sampler,
            "DerivationTreeNode": state.DerivationTreeNode,
            "deepcopy": copy.deepcopy,
            "random": random,
            "psutil": psutil,
            "torch": torch,
            "pickle": pickle,
            "makedirs": os.makedirs,
            "join": os.path.join,
            "recursive_constrained_smith_waterman_crossover": algorithm.recursive_constrained_smith_waterman_crossover,
        },
        {"Individual", "Population", "Evolver"},
    )
    evolution_class = next(
        node
        for node in sources["search_strategies/evolution.py"].body
        if isinstance(node, ast.ClassDef) and node.name == "Evolution"
    )
    step = next(
        node
        for node in evolution_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "step"
    )
    exec(
        compile(
            ast.Module(body=[step], type_ignores=[]),
            str(root / "search_strategies/evolution.py"),
            "exec",
        ),
        evolution.__dict__,
    )
    return types.SimpleNamespace(
        root=root,
        sources=sources,
        pcfg=pcfg,
        state=state,
        utility=utility,
        layers=layers,
        einspace=einspace,
        hnasbench201=hnasbench201,
        sampler=sampler,
        algorithm=algorithm,
        evolution=evolution,
    )


def limits(batch=None):
    return load().utility.Limiter(
        {
            "time": 60,
            "restart_time": 300,
            "max_id": 10000,
            "depth": 20,
            "memory": 65536,
            "memory_crossover": 65536,
            "individual_memory": 8196,
            "batch_pass_seconds": 10,
        },
        batch=batch,
    )


def tree(description, *, node_type=None, operation_type=None, limiter=None):
    reference = load()
    node_type = node_type or reference.state.DerivationTreeNode
    operation_type = operation_type or reference.state.Operation
    next_id = 0

    def build(item, parent=None):
        nonlocal next_id
        if isinstance(item, str):
            name, children = item, ()
        else:
            name, *children = item
        operation = operation_type(
            name, None, None, None, [], [], "terminal" if not children else "nonterminal", []
        )
        node = node_type(next_id, operation=operation, parent=parent, limiter=limiter)
        next_id += 1
        node.children = [build(child, node) for child in children]
        return node

    return build(description)


def chain(names):
    if len(names) == 1:
        return ("computation", names[0])
    middle = len(names) // 2
    return ("sequential", chain(names[:middle]), chain(names[middle:]))


def tree_record(node):
    return (node.id, node.operation.name, tuple(tree_record(child) for child in node.children))


def operation_record(operation):
    def dependencies(items):
        return tuple(
            tuple(item.id for item in group) if type(group) is list else group.id for group in items
        )

    return (
        operation.id,
        operation.op_type,
        operation.node1_id,
        operation.node2_id,
        operation.i,
        operation.j,
        operation.ii,
        operation.jj,
        float(operation.value),
        operation.i_swapped,
        operation.j_swapped,
        dependencies(operation.enabler_ops),
        dependencies(operation.disabler_ops),
    )
