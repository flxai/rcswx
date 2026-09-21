import copy
import math
import pickle
from types import SimpleNamespace

import pytest
import torch
from rcswx import (
    PCFG,
    Limiter,
    Reconstructor,
    gated_crossover,
    generation_step,
    raw_crossover,
    validated_crossover,
)
from rcswx.grammars import einspace

from tests.reference.corpus import input_params, operation_map
from tests.reference.original import limits, load, operation_record, tree_record
from tests.test_reference_construction import metadata, rng_record, seed


def compile_task(root):
    return torch.nn.Sequential(
        root.build(root),
        torch.nn.Flatten(1),
        torch.nn.Linear(math.prod(root.output_params["shape"][1:]), 3),
    )


def evaluate(root):
    model = compile_task(root)
    loss = torch.nn.functional.cross_entropy(model(torch.randn(2, 8, 8)), torch.tensor([0, 2]))
    loss.backward()
    return float(loss.detach())


def prepare(owned, right="linear(32)", rate=1.0):
    reference = load()
    module = einspace if owned else reference.einspace
    guard_type = Limiter if owned else reference.utility.Limiter
    pcfg_type = PCFG if owned else reference.pcfg.PCFG
    sampler_type = Reconstructor if owned else reference.sampler.Sampler
    guard = guard_type(
        limits().limits, batch=torch.zeros(2, 8, 8), compile_fn=compile_task, n_batch_passes=1
    )
    grammar = copy.deepcopy(module.grammar)
    choices = operation_map(module)
    identity_permutation = module.permute([0, 1, 2])
    choices[identity_permutation.name] = identity_permutation
    grammar["prerouting_fn"]["options"].append(identity_permutation)
    grammar["prerouting_fn"]["probs"].append(0.1)
    pcfg = pcfg_type(grammar, guard)
    sampler = sampler_type(pcfg, guard, "iterative")
    if right == "fractional":
        recipes = [
            ("routing", before, "computation", "relu", "identity")
            for before in ("perm(0,1,2)", "perm(0,2,1)")
        ]
    else:
        recipes = [
            ("sequential", "computation", name, "computation", "relu")
            for name in ("linear(16)", right)
        ]
    parents = tuple(
        sampler.sample(input_params((2, 8, 8)), operations=[choices[item] for item in recipe])
        for recipe in recipes
    )
    evolver = reference.evolution.Evolver(
        pcfg=pcfg,
        limiter=guard,
        crossover_strategy="recursive_constrained_smith_waterman",
        crossover_rate=rate,
        mutation_rate=0,
        selection_strategy="first",
    )
    return parents, guard, sampler, evolver


def report_record(report):
    return {
        key: tuple(map(operation_record, value))
        if key in ("crossover_operations", "crossover_all_operations")
        else value
        for key, value in report.items()
    }


def assert_rng_equal(actual, expected):
    assert actual[:2] == expected[:2]
    assert torch.equal(actual[2], expected[2])


@pytest.mark.parametrize(
    "right,ceiling,max_tries",
    [
        ("linear(16)", None, 100),
        ("linear(32)", None, 100),
        ("linear(32)", 16, 100),
        ("linear(32)", 8, 2),
    ],
)
def test_validation_reconstructs_redraws_and_preserves_retry_boundary(right, ceiling, max_tries):
    outcomes = []
    for owned in (False, True):
        seed()
        parents, guard, sampler, evolver = prepare(owned, right)
        input_before = parents[0].input_params
        widths = []

        def rebuild(child):
            child = sampler.re_id(child)
            width = child.output_params["shape"][-1]
            widths.append(width)
            if ceiling is not None and width > ceiling:
                raise MemoryError("Constructed network exceeds caller's width allowance")
            return child

        try:
            if owned:
                child, report = validated_crossover(
                    *parents, rebuild=rebuild, batch_shape=guard.batch_shape, max_tries=max_tries
                )
            else:
                evolver.re_id = rebuild
                child, report = evolver.recursive_constrained_smith_waterman_crossover(
                    *parents, max_tries=max_tries
                )
            result = metadata(child), report_record(report), child is parents[0]
        except Exception as error:
            result = type(error).__name__
        outcomes.append(
            (
                result,
                widths,
                parents[0].input_params is input_before,
                tuple(map(metadata, parents)),
                rng_record(),
            )
        )
    assert outcomes[0][:-1] == outcomes[1][:-1]
    assert_rng_equal(outcomes[1][-1], outcomes[0][-1])
    if ceiling == 8:
        assert outcomes[1][0] == "RuntimeError"
        assert len(outcomes[1][1]) == max_tries + 1
    elif ceiling == 16:
        assert outcomes[1][1][0] == 32 and outcomes[1][1][-1] == 16
    elif right == "linear(32)":
        assert outcomes[1][1] == [32]  # Different parent output shapes are not pre-filtered.
    else:
        assert outcomes[1][2] is False  # The reference raw no-op aliases parent1's inputs.


@pytest.mark.parametrize("failure", ["fractional_selector", "alignment_memory"])
def test_raw_failures_escape_the_inner_build_retry_handler(failure):
    for owned in (False, True):
        seed()
        parents, guard, sampler, evolver = prepare(
            owned, "fractional" if failure == "fractional_selector" else "linear(32)"
        )
        if failure == "alignment_memory":
            guard.limits["memory_crossover"] = 0
        attempts = []

        def rebuild(root):
            attempts.append(root)
            return sampler.re_id(root)

        before = rng_record()
        expected = IndexError if failure == "fractional_selector" else MemoryError
        with pytest.raises(expected):
            if owned:
                validated_crossover(*parents, rebuild=rebuild, batch_shape=guard.batch_shape)
            else:
                evolver.re_id = rebuild
                evolver.recursive_constrained_smith_waterman_crossover(*parents)
        assert attempts == []
        assert_rng_equal(rng_record(), before)


@pytest.mark.parametrize("rate", [0, 0.5, 1])
def test_python_rng_gate_and_no_crossover_alias(rate):
    outcomes = []
    for owned in (False, True):
        seed()
        parents, guard, sampler, evolver = prepare(owned, rate=rate)
        if owned:
            child, report = gated_crossover(
                *parents,
                crossover_rate=rate,
                rebuild=sampler.re_id,
                batch_shape=guard.batch_shape,
                limiter=guard,
            )
        else:
            child, report = evolver.crossover(*parents)
        outcomes.append(
            (
                metadata(child),
                report_record(report),
                child is parents[0],
                tuple(map(metadata, parents)),
                rng_record(),
            )
        )
    assert outcomes[0][:-1] == outcomes[1][:-1]
    assert_rng_equal(outcomes[1][-1], outcomes[0][-1])
    if rate == 0:
        assert outcomes[1][2] is True


def test_native_raw_operator_inside_unmodified_original_evolver(monkeypatch):
    outcomes = []
    for native in (False, True):
        seed()
        parents, guard, sampler, evolver = prepare(False)
        population = load().evolution.Population(
            [
                load().evolution.Individual(index, root=root, accuracy=index)
                for index, root in enumerate(parents)
            ]
        )
        if native:
            monkeypatch.setattr(
                load().evolution, "recursive_constrained_smith_waterman_crossover", raw_crossover
            )
        child, ancestry = evolver.evolve(population)
        normalized = {
            key: metadata(value) if key in ("parent1", "parent2") and value is not None else value
            for key, value in report_record(ancestry).items()
        }
        outcomes.append((metadata(child), normalized, rng_record()))
    assert outcomes[0][:-1] == outcomes[1][:-1]
    assert_rng_equal(outcomes[1][-1], outcomes[0][-1])


def test_outer_generation_reselects_and_saves_debug_parents(tmp_path):
    outcomes = []
    for owned in (False, True):
        seed()
        directory = tmp_path / ("native" if owned else "original")
        results_path = str(directory / "results.pkl")
        attempts = []
        current = SimpleNamespace(parent1=None, parent2=None)
        _, guard, _, _ = prepare(owned)

        def generate():
            attempt = len(attempts)
            parents, _, sampler, evolver = prepare(
                owned, "fractional" if attempt == 0 else "linear(32)"
            )
            current.parent1 = load().evolution.Individual(10, root=parents[0])
            current.parent2 = load().evolution.Individual(11, root=parents[1])
            attempts.append(tuple(map(tree_record, parents)))
            if owned:
                return validated_crossover(
                    *parents, rebuild=sampler.re_id, batch_shape=guard.batch_shape
                )
            return evolver.recursive_constrained_smith_waterman_crossover(*parents)

        if owned:
            accepted = generation_step(
                generate,
                evaluate,
                limiter=guard,
                n_tries=2,
                parents=lambda: (current.parent1, current.parent2),
                results_path=results_path,
                iteration=7,
            )
            root, reward, ancestry = accepted.root, accepted.reward, accepted.ancestry
        else:
            current.evolve = lambda population: generate()
            state = SimpleNamespace(
                limiter=guard,
                generational=False,
                population=load().evolution.Population([]),
                evolver=current,
                verbose=False,
                evaluation_fn=evaluate,
                n_tries=2,
                results_path=results_path,
                rewards=[],
                regularised=False,
                save_results=lambda iteration: None,
            )
            load().evolution.step(state, 7, "evolve")
            accepted = state.population[-1]
            root, reward, ancestry = accepted.root, accepted.accuracy, accepted.ancestry
        debug = []
        for side in (1, 2):
            with (directory / "debug" / f"parent{side}_iter7_id{9 + side}.pkl").open("rb") as saved:
                debug.append(tree_record(pickle.load(saved)))
        outcomes.append(
            (metadata(root), reward, report_record(ancestry), attempts, debug, rng_record())
        )
    assert len(outcomes[1][3]) == 2
    assert outcomes[0][:-1] == outcomes[1][:-1]
    assert_rng_equal(outcomes[1][-1], outcomes[0][-1])
