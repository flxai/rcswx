"""Observable worker policy, deterministic plans, and native host lifetimes."""

import json
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from rcswx import Alignment, Architecture, NativeRng, _core, apply_edits, edit_path

PARALLEL = pytest.mark.skipif(not _core.PARALLEL_CAPABLE, reason="serial-only native build")


def parents(size=48, *, nested=False):
    def build(changed):
        items = [
            ("computation", (f"linear({16 + 2 * index + int(changed and index == size // 2)})",))
            for index in range(size)
        ]
        if nested:
            tree = items[-1]
            for leaf in reversed(items[:-1]):
                tree = ("routing", ("identity",), ("sequential", leaf, tree), ("identity",))
        else:
            items = [("routing", ("identity",), leaf, ("identity",)) for leaf in items]
            while len(items) > 1:
                items = [
                    ("sequential", *items[index : index + 2]) for index in range(0, len(items), 2)
                ]
            tree = items[0]
        return Architecture.from_tree(tree)

    return build(False), build(True)


def prepared(size=48):
    first, second = parents(size, nested=True)
    return _core.prepare_architectures(first.to_json(), second.to_json())


def legacy_leaf(identifier, name):
    return SimpleNamespace(
        id=identifier, operation=SimpleNamespace(name=name), children=[], limiter=None
    )


@pytest.fixture
def two_workers():
    report = json.loads(prepared(2).analyze("capacity", workers=2).execution_json())
    if report["worker_limit"] < 2:
        pytest.skip("the native pool has only one worker")


@pytest.mark.parametrize(
    "workers,error",
    [
        (True, TypeError),
        (False, TypeError),
        (1.0, TypeError),
        ("2", TypeError),
        (None, TypeError),
        (0, ValueError),
        (-2, ValueError),
        (-(1 << 100), ValueError),
        (1 << 100, OverflowError),
    ],
)
def test_invalid_workers_precede_legacy_preparation(workers, error):
    for factory in (edit_path, Alignment):
        first, second = legacy_leaf(100, "identity"), legacy_leaf(200, "relu")
        with pytest.raises(error):
            factory(first, second, workers=workers)
        assert (first.id, second.id) == (100, 200)


@pytest.mark.parametrize(
    "workers,error", [(True, TypeError), (0, ValueError), (1 << 100, OverflowError)]
)
def test_direct_native_calls_do_not_bypass_worker_validation(workers, error):
    snapshot = prepared(2)
    with pytest.raises(error):
        snapshot.analyze("invalid", workers=workers)
    with pytest.raises(error):
        _core.recursive_align([], [], workers=workers)


def test_serial_only_build_rejects_parallel_before_mutation():
    if _core.PARALLEL_CAPABLE:
        pytest.skip("this assertion targets the separately built serial artifact")
    for workers in (2, -1):
        first, second = legacy_leaf(100, "identity"), legacy_leaf(200, "relu")
        with pytest.raises(RuntimeError):
            edit_path(first, second, workers=workers)
        assert (first.id, second.id) == (100, 200)
        with pytest.raises(RuntimeError):
            prepared(2).analyze("unsupported", workers=workers)


@PARALLEL
@pytest.mark.parametrize("collapse", [False, True])
@pytest.mark.parametrize("workers", [2, 4, -1])
@pytest.mark.parametrize("nested", [False, True])
def test_parallel_preserves_complete_ordered_plans_and_consumers(collapse, workers, nested):
    pair = parents(nested=nested)
    serial = edit_path(*pair, collapse_corners=collapse, workers=1)
    parallel = edit_path(*pair, collapse_corners=collapse, workers=workers)
    assert parallel.distance == serial.distance == 0.25
    assert parallel._native.paths_json() == serial._native.paths_json()
    assert parallel.stats == serial.stats
    for indices in ([], "1" * len(serial.nontrivial_ops)):
        assert (
            apply_edits(parallel, parallel.select(indices)).to_json()
            == apply_edits(serial, serial.select(indices)).to_json()
        )
    first_rng, second_rng = NativeRng(42), NativeRng(42)
    first_sample, second_sample = serial.sample(rng=first_rng), parallel.sample(rng=second_rng)
    assert (first_sample.indices, first_sample.cost) == (second_sample.indices, second_sample.cost)
    assert first_rng.state() == second_rng.state()
    report = parallel.execution
    capacity = report["pool_capacity"]
    assert report["worker_limit"] == (capacity if workers == -1 else min(workers, capacity))
    assert report["peak_jobs"] <= report["worker_limit"]
    if capacity > 1 and nested:
        assert report["parallel_cells"] > 0
        assert report["parallel_rounds"] > 0


@PARALLEL
@pytest.mark.parametrize("failure", ["false", "exception"])
def test_callback_failure_preserves_exception_and_executor_reuse(failure, two_workers):
    snapshot = prepared()
    coordinator = threading.get_ident()
    marker = LookupError("callback marker")
    calls = 0

    def check():
        nonlocal calls
        assert threading.get_ident() == coordinator
        calls += 1
        if calls == 3:
            if failure == "exception":
                raise marker
            return False
        return True

    with pytest.raises(LookupError if failure == "exception" else MemoryError) as raised:
        snapshot.analyze("cancelled", workers=2, memory_check=check)
    if failure == "exception":
        assert raised.value is marker
    assert calls == 3
    recovered = snapshot.analyze("recovered", workers=2)
    serial = snapshot.analyze("serial", workers=1)
    assert recovered.paths_json() == serial.paths_json()


@PARALLEL
def test_callback_reentry_allows_serial_but_rejects_parallel_before_preparation():
    snapshot = prepared()
    checked = False

    def check():
        nonlocal checked
        if not checked:
            checked = True
            assert snapshot.analyze("nested-serial", workers=1).distance == 0.25
            with pytest.raises(RuntimeError):
                snapshot.analyze("nested-parallel", workers=2)
            first, second = legacy_leaf(100, "identity"), legacy_leaf(200, "relu")
            with pytest.raises(RuntimeError):
                edit_path(first, second, workers=2)
            assert (first.id, second.id) == (100, 200)
        return True

    plan = snapshot.analyze("outer", workers=2, memory_check=check)
    assert checked
    assert plan.distance == 0.25


@PARALLEL
def test_concurrent_cancellation_does_not_cancel_another_call(two_workers):
    snapshots = [prepared(48), prepared(56)]
    expected = snapshots[1].analyze("serial", workers=1).paths_json()
    entered = threading.Barrier(2)

    def analyze(index):
        calls = 0

        def check():
            nonlocal calls
            calls += 1
            if calls == 1:
                entered.wait(timeout=15)
            return index != 0 or calls < 3

        return snapshots[index].analyze(str(index), workers=2, memory_check=check)

    with ThreadPoolExecutor(max_workers=2) as callers:
        cancelled = callers.submit(analyze, 0)
        successful = callers.submit(analyze, 1)
        with pytest.raises(MemoryError):
            cancelled.result(timeout=30)
        plan = successful.result(timeout=30)
    assert plan.paths_json() == expected
    assert json.loads(plan.execution_json())["peak_jobs"] <= 2


@PARALLEL
@pytest.mark.skipif(not hasattr(os, "fork"), reason="POSIX fork behavior")
def test_forked_child_fails_fast_for_parallel_and_still_runs_serial():
    script = """
import os
import signal
from rcswx import Architecture, edit_path
first = Architecture.from_tree(('computation', ('identity',)))
second = Architecture.from_tree(('computation', ('relu',)))
edit_path(first, second, workers=2)
pid = os.fork()
if pid == 0:
    signal.alarm(10)
    try:
        try:
            edit_path(first, second, workers=2)
        except RuntimeError:
            pass
        else:
            os._exit(2)
        assert edit_path(first, second, workers=1).distance == 0.5
    except BaseException:
        os._exit(3)
    os._exit(0)
_, status = os.waitpid(pid, 0)
assert os.waitstatus_to_exitcode(status) == 0, status
assert edit_path(first, second, workers=2).distance == 0.5
"""
    subprocess.run([sys.executable, "-I", "-c", script], check=True, timeout=20)


@PARALLEL
@pytest.mark.skipif(sys.platform != "linux", reason="Linux thread accounting")
def test_default_and_explicit_serial_calls_do_not_create_pool_threads():
    script = """
import os
from rcswx import Architecture, edit_path
first = Architecture.from_tree(('computation', ('identity',)))
second = Architecture.from_tree(('computation', ('relu',)))
before = len(os.listdir('/proc/self/task'))
for options in ({}, {'workers': 1}):
    plan = edit_path(first, second, **options)
    assert plan.distance == 0.5
    assert len(os.listdir('/proc/self/task')) == before
parallel = edit_path(first, second, workers=2)
assert len(os.listdir('/proc/self/task')) > before
assert parallel.distance == plan.distance
"""
    subprocess.run([sys.executable, "-I", "-c", script], check=True, timeout=20)


@PARALLEL
@pytest.mark.skipif(os.name != "posix", reason="POSIX process signals")
@pytest.mark.parametrize("with_callback", [False, True])
def test_sigint_survives_detached_native_execution(with_callback, two_workers):
    script = """
import json
import os
import signal
import sys
import threading
import time
from rcswx import _core
first, second, recovery_first, recovery_second = json.loads(sys.stdin.read())
snapshot = _core.prepare_architectures(first, second)
with_callback = sys.argv[1] == 'True'
started = threading.Event()
polls = 0
def check():
    global polls
    polls += 1
    if polls == 2:
        started.set()
    return True
def interrupt():
    assert started.wait(10)
    time.sleep(0.01)
    os.kill(os.getpid(), signal.SIGINT)
thread = threading.Thread(target=interrupt, daemon=True)
thread.start()
if not with_callback:
    started.set()
try:
    snapshot.analyze('interrupted', workers=2, memory_check=check if with_callback else None)
except KeyboardInterrupt:
    thread.join(timeout=2)
    assert not thread.is_alive()
else:
    raise AssertionError('alignment returned a plan after SIGINT')
recovery = _core.prepare_architectures(recovery_first, recovery_second)
assert recovery.analyze('recovered', workers=2).distance == 0.25
"""
    pair = (*parents(64, nested=True), *parents(2))
    subprocess.run(
        [sys.executable, "-I", "-c", script, str(with_callback)],
        input=json.dumps([value.to_json() for value in pair]),
        text=True,
        check=True,
        timeout=30,
    )
