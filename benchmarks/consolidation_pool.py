"""Forked, CPU-pinned benchmark lanes with durable per-request evidence."""

import hashlib
import heapq
import json
import multiprocessing
import os
import signal
import sys
import time
import traceback
from contextlib import ExitStack
from multiprocessing.connection import wait

ASSIGNMENT = "request-sha256-v1"


def assigned_cpu(pair_id, phase, repeat_id, sampler_seed, cpus):
    key = json.dumps([pair_id, phase, repeat_id, sampler_seed], separators=(",", ":"))
    index = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big") % len(cpus)
    return cpus[index]


def _lane(requests, execute, cpus, cpu, directory, connection, start):
    try:
        os.setsid()
        os.sched_setaffinity(0, {cpu})
        with (directory / f"cpu-{cpu}.log").open("x") as log:
            sys.stdout = sys.stderr = log
            connection.send(("ready", None))
            start.wait()
            with (directory / f"cpu-{cpu}.jsonl").open("x") as stream:
                for request in requests:
                    if (
                        assigned_cpu(
                            request["pair"]["id"],
                            request["phase"],
                            request["repeat_id"],
                            request["sampler_seed"],
                            cpus,
                        )
                        != cpu
                    ):
                        continue
                    row = execute(request, cpu)
                    row.update(request_index=request["request_index"], worker_cpu=cpu)
                    stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
                    stream.flush()
                    del row
                    connection.send(("row", None))
            connection.send(("done", None))
    except BaseException:
        connection.send(("error", traceback.format_exc()))
        raise
    finally:
        connection.close()


def _stop_lanes(lanes):
    # Each owned lane is a session leader; its bounded request child shares that
    # process group. Killing just the coordinator would leave an active request.
    for process, _ in lanes:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            if process.is_alive():
                process.terminate()
    deadline = time.monotonic() + 5
    for process, _ in lanes:
        process.join(max(0, deadline - time.monotonic()))
    for process, _ in lanes:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            if process.is_alive():
                process.kill()
        process.join()


def _merge_rows(requests, cpus, directory, output):
    with ExitStack() as stack:
        streams = [stack.enter_context((directory / f"cpu-{cpu}.jsonl").open()) for cpu in cpus]
        rows = heapq.merge(
            *(map(json.loads, stream) for stream in streams),
            key=lambda row: row["request_index"],
        )
        count = 0
        with output.open("x") as stream:
            for row in rows:
                if row["request_index"] != count or count >= len(requests):
                    raise ValueError("Duplicate, missing, or unordered parallel request")
                request = requests[count]
                expected = (
                    request["pair"]["id"],
                    request["phase"],
                    request["repeat_id"],
                    request["sampler_seed"],
                )
                actual = tuple(row[key] for key in ("pair", "phase", "repeat_id", "sampler_seed"))
                if actual != expected or row["worker_cpu"] != assigned_cpu(*expected, cpus):
                    raise ValueError("Parallel row disagrees with its scheduled request")
                stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
                count += 1
        if count != len(requests):
            raise ValueError("Incomplete parallel request coverage")


def run_pool(requests, execute, cpus, output):
    """Run each request once on its stable CPU; retain partial lane files on failure."""
    if not cpus or len(set(cpus)) != len(cpus):
        raise ValueError("Parallel CPUs must be nonempty and distinct")
    if any(request["request_index"] != index for index, request in enumerate(requests)):
        raise ValueError("Parallel requests must have contiguous ordered indices")
    directory = output / "lanes"
    directory.mkdir()
    context = multiprocessing.get_context("fork")
    start = context.Event()
    lanes = []
    ready, done, exited, closed = set(), set(), set(), set()
    completed = 0
    started = time.monotonic()
    succeeded = False
    try:
        for cpu in cpus:
            receiver, sender = context.Pipe(duplex=False)
            process = context.Process(
                target=_lane,
                args=(requests, execute, cpus, cpu, directory, sender, start),
            )
            try:
                process.start()
            except BaseException:
                receiver.close()
                raise
            finally:
                sender.close()
            lanes.append((process, receiver))
        while len(exited) != len(lanes):
            handles = [receiver for index, (_, receiver) in enumerate(lanes) if index not in closed]
            handles.extend(
                process.sentinel for index, (process, _) in enumerate(lanes) if index not in exited
            )
            wait(handles)
            for index, (process, receiver) in enumerate(lanes):
                while index not in closed and receiver.poll():
                    try:
                        kind, detail = receiver.recv()
                    except EOFError:
                        closed.add(index)
                        break
                    if kind == "ready":
                        ready.add(index)
                    elif kind == "row":
                        completed += 1
                        if completed % 100 == 0 or completed == len(requests):
                            print(
                                json.dumps(
                                    {
                                        "completed": completed,
                                        "planned": len(requests),
                                        "seconds": time.monotonic() - started,
                                    }
                                ),
                                flush=True,
                            )
                    elif kind == "done":
                        done.add(index)
                    else:
                        raise RuntimeError(f"Parallel lane CPU {cpus[index]} failed:\n{detail}")
                if process.exitcode is not None and index not in exited:
                    process.join()
                    if process.exitcode != 0 or index not in done:
                        raise RuntimeError(
                            f"Parallel lane CPU {cpus[index]} exited without completing "
                            f"(exit {process.exitcode})"
                        )
                    exited.add(index)
            if len(ready) == len(lanes):
                start.set()
        if completed != len(requests):
            raise ValueError("Parallel progress does not cover every request")
        _merge_rows(requests, cpus, directory, output / "rows.jsonl")
        succeeded = True
    finally:
        if not succeeded:
            _stop_lanes(lanes)
        for _, receiver in lanes:
            receiver.close()
