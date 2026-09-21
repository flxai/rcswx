import copy
import functools
import math
import random

import numpy as np
import pytest
import torch
from rcswx.recursive import Alignment

from tests.reference.corpus import build_case, construction_cases
from tests.reference.original import limits, load, operation_record, tree_record

CASES = construction_cases()


@pytest.fixture(scope="module", autouse=True)
def single_threaded_construction():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def seed():
    random.seed(71)
    np.random.seed(71)
    torch.manual_seed(71)


def rng_record():
    numpy_state = np.random.get_state()
    return (
        random.getstate(),
        (numpy_state[0], numpy_state[1].tobytes(), *numpy_state[2:]),
        torch.get_rng_state(),
    )


def metadata(root):
    nodes = root.serialise()
    identities = {id(node): index for index, node in enumerate(nodes)}
    containers = {}

    def value(item):
        if isinstance(item, dict):
            identity = containers.setdefault(id(item), len(containers))
            return "dict", identity, tuple((key, value(child)) for key, child in item.items())
        if isinstance(item, (list, tuple)):
            return type(item).__name__, tuple(value(child) for child in item)
        if isinstance(item, functools.partial):
            return item.func.__name__, item.args, item.keywords
        if callable(item):
            return item.__name__
        return item

    return tuple(
        (
            node.id,
            node.level,
            node.depth,
            identities.get(id(node.parent)),
            tuple(identities[id(child)] for child in node.children),
            node.operation.name,
            node.operation.type,
            tuple(node.operation.child_levels),
            value(node.input_params),
            value(node.output_params),
            value(node.operation.build),
            value(node.operation.infer),
            value(node.operation.valid),
            value(node.operation.inherit),
            value(node.operation.give_back),
        )
        for node in nodes
    )


def execute(case, owned):
    seed()
    root, model = build_case(case, owned=owned)
    batch = torch.linspace(-1, 1, steps=math.prod(case[2])).reshape(case[2]).requires_grad_()
    result = model(batch)
    result.sum().backward()
    return root, model, result, batch.grad, rng_record()


@pytest.mark.parametrize("name", CASES)
def test_registered_construction_forward_gradients_metadata_and_rng(name):
    case = CASES[name]
    try:
        expected = execute(case, False)
    except Exception as error:
        with pytest.raises(type(error)):
            execute(case, True)
        return
    actual = execute(case, True)
    old_root, old_model, old_output, old_gradient, old_rng = expected
    new_root, new_model, new_output, new_gradient, new_rng = actual
    assert metadata(new_root) == metadata(old_root)
    torch.testing.assert_close(new_output, old_output, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(new_gradient, old_gradient, rtol=1e-5, atol=1e-6)
    assert new_rng[:2] == old_rng[:2]
    assert torch.equal(new_rng[2], old_rng[2])
    assert list(new_model.state_dict()) == list(old_model.state_dict())
    for name, state in old_model.state_dict().items():
        assert torch.equal(new_model.state_dict()[name], state)
    old_parameters = dict(old_model.named_parameters())
    new_parameters = dict(new_model.named_parameters())
    assert new_parameters.keys() == old_parameters.keys()
    for name, parameter in old_parameters.items():
        torch.testing.assert_close(new_parameters[name].grad, parameter.grad, rtol=1e-5, atol=1e-6)
        assert (
            new_parameters[name].untyped_storage().data_ptr()
            != parameter.untyped_storage().data_ptr()
        )


@pytest.mark.parametrize("name", CASES)
def test_every_registered_genotype_alignment_outcome(name):
    case = CASES[name]
    seed()
    try:
        old_root, _ = build_case(case)
    except Exception as error:
        with pytest.raises(type(error)):
            build_case(case, owned=True)
        return
    new_root, _ = build_case(case, owned=True)
    old_parents = (old_root, copy.deepcopy(old_root))
    new_parents = (new_root, copy.deepcopy(new_root))
    try:
        old = load().algorithm.AlignmentMatrixRecursive(*old_parents, limiter=limits())
    except Exception as error:
        with pytest.raises(type(error)):
            Alignment(*new_parents, limiter=limits())
    else:
        new = Alignment(*new_parents, limiter=limits())
        assert new.distance == old.distance
        assert tuple(map(operation_record, new.operations)) == tuple(
            map(operation_record, old.operations)
        )
    assert tuple(map(tree_record, new_parents)) == tuple(map(tree_record, old_parents))
