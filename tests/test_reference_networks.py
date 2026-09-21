"""Compound networks catch interactions absent from single-production tests."""

import math

import pytest
import torch
from rcswx import Alignment, raw_crossover, validated_crossover

from tests.reference.networks import network_pairs, prepare_pair
from tests.reference.original import load, operation_record, tree_record
from tests.test_reference_construction import metadata, rng_record, seed
from tests.test_reference_pipeline import assert_rng_equal, report_record

CASES = network_pairs()


@pytest.fixture(scope="module", autouse=True)
def single_threaded_networks():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def observe_model(root, shape):
    model = root.build(root)
    batch = torch.linspace(-1, 1, steps=math.prod(shape)).reshape(shape).requires_grad_()
    output = model(batch)
    assert torch.isfinite(output).all()
    output.square().mean().backward()
    return {
        "output": output.detach(),
        "input_gradient": batch.grad,
        "state": model.state_dict(),
        "parameter_gradients": {
            name: parameter.grad for name, parameter in model.named_parameters()
        },
    }


@pytest.mark.parametrize("name", CASES)
def test_compound_parents_execute_with_matching_state_gradients_and_metadata(name):
    observations = []
    for owned in (False, True):
        seed()
        parents, _, _ = prepare_pair(CASES[name], owned)
        models = [observe_model(root, CASES[name].shape) for root in parents]
        observations.append((models, tuple(map(metadata, parents)), rng_record()))
    expected, actual = observations
    torch.testing.assert_close(actual[0], expected[0], rtol=1e-5, atol=1e-6)
    assert actual[1] == expected[1]
    assert_rng_equal(actual[2], expected[2])


@pytest.mark.parametrize("name", CASES)
@pytest.mark.parametrize("reverse", (False, True), ids=("forward", "reverse"))
@pytest.mark.parametrize("phase", ("align", "raw"))
def test_compound_alignment_and_raw_crossover_preserve_source_outcomes(name, reverse, phase):
    observations = []
    for owned in (False, True):
        seed()
        parents, _, guard = prepare_pair(CASES[name], owned)
        if reverse:
            parents = tuple(reversed(parents))
        reference = load().algorithm
        try:
            if phase == "align":
                alignment_type = Alignment if owned else reference.AlignmentMatrixRecursive
                alignment = alignment_type(*parents, limiter=guard)
                paths = alignment.paths if owned else alignment.matrix[-1][-1].paths
                result = (
                    alignment.distance,
                    tuple(tuple(map(operation_record, path)) for path in paths),
                    tuple(map(operation_record, alignment.operations)),
                )
            else:
                crossover = (
                    raw_crossover
                    if owned
                    else reference.recursive_constrained_smith_waterman_crossover
                )
                child, selected, operations, *distances = crossover(*parents, limiter=guard)
                result = (
                    tree_record(child),
                    tuple(map(operation_record, selected)),
                    tuple(map(operation_record, operations)),
                    distances,
                    tuple(child is parent for parent in parents),
                )
            outcome = "returned", result
        except Exception as error:
            outcome = "raised", type(error).__name__
        observations.append((outcome, tuple(map(metadata, parents)), rng_record()))
    expected, actual = observations
    assert actual[:2] == expected[:2]
    assert_rng_equal(actual[2], expected[2])


@pytest.mark.parametrize("name", CASES)
@pytest.mark.parametrize("reverse", (False, True), ids=("forward", "reverse"))
def test_compound_validation_and_accepted_offspring_training_match_source(name, reverse):
    observations = []
    for owned in (False, True):
        seed()
        parents, builder, guard = prepare_pair(CASES[name], owned)
        if reverse:
            parents = tuple(reversed(parents))
        try:
            if owned:
                child, report = validated_crossover(
                    *parents,
                    rebuild=builder.re_id,
                    batch_shape=guard.batch_shape,
                    limiter=guard,
                    max_tries=3,
                )
            else:
                child, report = builder.recursive_constrained_smith_waterman_crossover(
                    *parents, max_tries=3
                )
        except Exception as error:
            outcome, model = ("raised", type(error).__name__), None
        else:
            outcome = "returned", metadata(child), report_record(report)
            model = observe_model(child, CASES[name].shape)
        observations.append((outcome, model, tuple(map(metadata, parents)), rng_record()))
    expected, actual = observations
    assert actual[0] == expected[0]
    torch.testing.assert_close(actual[1], expected[1], rtol=1e-5, atol=1e-6)
    assert actual[2] == expected[2]
    assert_rng_equal(actual[3], expected[3])
