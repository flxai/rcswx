import itertools

import pytest
from rcswx import CancellationToken, _core

from tests.reference.sampling import distribution, draws
from tests.test_alignment import IDENTITY, LINEAR4, LINEAR8, RELU, native_architecture, witness


def test_native_cdf_and_words_agree_with_independent_scalar_reference():
    limits, token = _core.Limits(), CancellationToken()
    alphabet = (IDENTITY, LINEAR4, LINEAR8)
    sequences = [sequence for n in (1, 2) for sequence in itertools.product(alphabet, repeat=n)]
    for left, right in itertools.product(sequences, repeat=2):
        path = _core.align(native_architecture(left), native_architecture(right), limits, token)
        history = witness(path)
        expected = distribution(left, right, history)
        for seed in (0, 42, (1 << 64) - 1):
            sampler = _core.Sampler(path, seed, limits, token)
            assert sampler.stats[1] == len(expected)
            assert sampler.stats[3] == expected[-1][1]
            assert [sampler.draw() for _ in range(32)] == draws(left, right, history, seed, 32)


def test_mask_multiplicity_is_not_collapsed_to_distinct_architectures():
    left = (IDENTITY, IDENTITY, RELU, RELU, LINEAR4, LINEAR4, RELU, RELU, LINEAR4, LINEAR4)
    right = left[2:] + left[:2]
    limits, token = _core.Limits(), CancellationToken()
    path = _core.align(native_architecture(left), native_architecture(right), limits, token)
    history = witness(path)
    expected = distribution(left, right, history)
    from tests.reference.ordered import projection

    architectures = [projection(left, right, history, mask)[0] for mask, _ in expected]
    assert len(set(architectures)) < len(expected)
    sampler = _core.Sampler(path, 42, limits, token)
    assert sampler.stats[1] == len(expected)
    assert sampler.stats[3] == expected[-1][1]
    assert [sampler.draw() for _ in range(128)] == draws(left, right, history, 42, 128)


def test_retained_cdf_and_projection_share_one_memory_budget():
    token = CancellationToken()
    path = _core.align(
        native_architecture((IDENTITY,)),
        native_architecture((LINEAR8, LINEAR4)),
        _core.Limits(),
        token,
    )
    probe = _core.Sampler(path, 42, _core.Limits(), token)
    limits = _core.Limits(max_core_bytes=probe.stats[2])
    sampler = _core.Sampler(path, 42, limits, token)
    with pytest.raises(_core.CoreError) as failure:
        sampler.propose(limits, token)
    assert failure.value.args[0:2] == ("budget", "max_core_bytes")
