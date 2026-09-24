"""A failed coordinator must not leave request grandchildren running."""

import multiprocessing
import os
import time

import psutil
import pytest

from benchmarks.consolidation_pool import run_pool


def test_lane_failure_stops_other_lanes_and_request_descendants(tmp_path):
    cpus = sorted(os.sched_getaffinity(0))[:2]
    if len(cpus) < 2:
        pytest.skip("Needs two allowed CPUs to exercise sibling-lane cancellation")
    barrier = multiprocessing.get_context("fork").Barrier(2)
    requests = [
        {
            "request_index": index,
            "pair": {"id": str(index)},
            "phase": "distance",
            "repeat_id": 0,
            "sampler_seed": 12,
        }
        for index in range(24)
    ]

    def interrupted(request, cpu):
        child = os.fork()
        if child == 0:
            time.sleep(60)
            os._exit(0)
        (tmp_path / f"child-{cpu}.pid").write_text(str(child))
        barrier.wait(timeout=10)
        if cpu == cpus[0]:
            raise RuntimeError("Intentional coordinator failure")
        time.sleep(60)
        raise AssertionError("Sibling lane was not cancelled")

    try:
        with pytest.raises(RuntimeError):
            run_pool(requests, interrupted, cpus, tmp_path)
        descendants = [int(path.read_text()) for path in tmp_path.glob("child-*.pid")]
        assert len(descendants) == 2
        for pid in descendants:
            try:
                assert psutil.Process(pid).status() in (psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD)
            except psutil.NoSuchProcess:
                pass
    finally:
        # Keep a broken implementation from leaking this test's sleeping children.
        for path in tmp_path.glob("child-*.pid"):
            try:
                psutil.Process(int(path.read_text())).kill()
            except psutil.NoSuchProcess:
                pass
