import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch
from rcswx import (
    BudgetExceeded,
    CancellationToken,
    Cancelled,
    Limits,
    TensorSpec,
    _core,
    crossover,
    crossover_with_report,
    edit_path,
    prepare,
)
from torch import nn


def test_retained_path_and_revalidated_inputs_share_the_core_budget():
    spec = TensorSpec((2, 4))
    left, right = nn.Linear(4, 4), nn.Linear(4, 4)
    a, b = (prepare(model, input_spec=spec) for model in (left, right))
    path = edit_path(a, b)
    base = crossover_with_report(a, b, path=path, seed=42).report["proposal"]["charged_bytes"]
    c, d = (prepare(model, input_spec=spec) for model in (left, right))
    extra = c.native.owned_bytes + d.native.owned_bytes
    full = crossover_with_report(c, d, path=path, seed=42)
    assert full.report["proposal"]["charged_bytes"] >= base + extra
    for inputs in ((c, d), (left, right)):
        with pytest.raises(BudgetExceeded) as failure:
            crossover(*inputs, path=path, seed=42, limits=Limits(max_core_bytes=base))
        assert failure.value.resource == "max_core_bytes"
    with pytest.raises(BudgetExceeded) as failure:
        crossover(left, right, path=path, limits=Limits(max_core_bytes=path.native.owned_bytes + 1))
    assert failure.value.resource == "max_core_bytes"


def test_shared_prepared_handles_are_not_charged_twice():
    spec = TensorSpec((2, 4))
    a, b = (prepare(nn.Identity(), input_spec=spec) for _ in range(2))
    bound = edit_path(a, a).native.stats["charged_bytes"]
    assert edit_path(a, a, limits=Limits(max_core_bytes=bound)).cost_ticks == 0
    with pytest.raises(BudgetExceeded):
        edit_path(a, b, limits=Limits(max_core_bytes=bound))


def test_candidate_container_depth_is_budgeted_before_building():
    limits = Limits(max_depth=1)
    architecture = prepare(nn.Identity(), input_spec=TensorSpec((2, 4)), limits=limits)
    with pytest.raises(BudgetExceeded) as failure:
        crossover(architecture, architecture, limits=limits, seed=42)
    assert failure.value.resource == "max_depth"
    assert failure.value.observed == 2


def test_cached_paths_cannot_bypass_stricter_request_limits_or_cancellation():
    a = prepare(nn.Sequential(nn.Identity(), nn.ReLU()), input_spec=TensorSpec((2, 4)))
    path = edit_path(a, a)
    for limits, resource in (
        (Limits(max_nodes=1), "max_nodes"),
        (Limits(max_rank=1), "max_rank"),
        (Limits(max_predecessors=1), "max_predecessors"),
        (Limits(deadline_seconds=1e-30), "deadline_seconds"),
    ):
        with pytest.raises(BudgetExceeded) as failure:
            crossover(a, a, path=path, limits=limits)
        assert failure.value.resource == resource
    token = CancellationToken()
    token.cancel()
    with pytest.raises(Cancelled):
        crossover(a, a, path=path, cancel=token)


def test_mask_enumeration_releases_interpreter_and_observes_cancellation():
    limits = _core.Limits(max_edits=22, max_masks=1 << 22)
    left = _core.Architecture([(1, 4, True, 0.0, None, False)] * 22, limits)
    right = _core.Architecture([(1, 8, True, 0.0, None, False)] * 22, limits)
    token = CancellationToken()
    path = _core.align(left, right, limits, token)
    gate = threading.Event()
    worker = threading.Thread(target=lambda: (gate.wait(), token.cancel()))
    interval = sys.getswitchinterval()
    worker.start()
    try:
        sys.setswitchinterval(1000)
        gate.set()
        with pytest.raises(_core.CoreError) as failure:
            _core.Sampler(path, 42, limits, token)
        assert failure.value.args[0] == "cancelled"
    finally:
        sys.setswitchinterval(interval)
        gate.set()
        worker.join(timeout=5)
    assert not worker.is_alive()


def test_parallel_prepared_requests_are_seeded_not_schedule_dependent():
    spec = TensorSpec((2, 4))
    a, b = (
        prepare(model, input_spec=spec)
        for model in (
            nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 4)),
            nn.Sequential(nn.Linear(4, 16), nn.Linear(16, 4)),
        )
    )
    path = edit_path(a, b)
    state = torch.get_rng_state()

    def run(seed):
        return crossover(a, b, path=path, seed=seed)

    serial = list(map(run, range(16)))
    with ThreadPoolExecutor(max_workers=4) as pool:
        concurrent = list(pool.map(run, range(16)))
    for left, right in zip(serial, concurrent, strict=True):
        assert (
            prepare(left, input_spec=spec).architecture_key
            == prepare(right, input_spec=spec).architecture_key
        )
        assert all(
            torch.equal(a, b) for a, b in zip(left.parameters(), right.parameters(), strict=True)
        )
    assert torch.equal(state, torch.get_rng_state())
