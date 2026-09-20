import json
from pathlib import Path

import pytest
from rcswx import _core
from rcswx.policies import Limits, resolve_seed, seed_streams


@pytest.mark.parametrize(
    "vector", json.loads((Path(__file__).parent / "fixtures/seed_vectors.json").read_text())
)
def test_seed_domain_vectors(vector):
    seed, domain = vector["seed"], vector["domain"]
    expected = bytes.fromhex(vector["hex"])
    assert bytes(_core.derive_seed(seed, domain.encode())) == expected
    selection, initialization = seed_streams(seed)
    if domain == "selection":
        assert selection == expected
    else:
        assert initialization == int.from_bytes(expected[:8], "little")


@pytest.mark.parametrize("seed", [-1, 1 << 64, True, 0.5])
def test_seed_rejects_non_u64(seed):
    with pytest.raises(ValueError):
        resolve_seed(seed)


@pytest.mark.parametrize("deadline", [float("inf"), float("nan"), 0, True])
def test_deadline_must_be_finite_positive(deadline):
    with pytest.raises(ValueError):
        Limits(deadline_seconds=deadline)
