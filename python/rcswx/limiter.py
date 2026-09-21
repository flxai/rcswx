# Derived from einsearch 7e713c7951397a6b12bd57638409774a74381746.
# Copyright (c) 2024 Adri Gómez Martín, Felix Möller, Linus Ericsson, Aaron Klein.
# Distributed under the MIT license; see LICENSE.einsearch.
"""Reference runtime resource checks, separate from alignment and selection."""

import ctypes
from functools import reduce
from time import perf_counter as time

import psutil
import torch


class Timer:
    def __init__(self):
        self.start_time = None
        self.end_time = None

    def start(self):
        self.start_time = time()

    def stop(self):
        self.end_time = time()

    def __call__(self):
        current_time = time()
        duration = current_time - self.start_time
        return duration

    def __str__(self):
        return f"Timer(start_time={self.start_time:.2f}, duration={self():.2f})"


class Limiter:
    def __init__(self, limits, batch=None, compile_fn=None, n_batch_passes=5):
        self.limits = limits
        self.timer = Timer()
        self.memory_checkpoint = None
        if batch is not None:
            self.batch_shape = batch.shape
            self.device = batch.device
        else:
            self.batch_shape = None
            self.device = None
        self.compile_fn = compile_fn
        self.n_batch_passes = n_batch_passes
        print(f"Limiter({self.limits})")

    def check(self, node, verbose=False):
        """
        Check if the limits have been reached.
        """
        duration = self.timer()
        self.memory = psutil.Process().memory_info().rss / (1024 * 1024)
        self.diff = (
            self.memory - self.memory_checkpoint if self.memory_checkpoint is not None else 0
        )
        if duration >= self.limits["restart_time"]:
            if verbose:
                print(f"Restarting sampling after {duration:.2f} seconds")
            raise RuntimeError("Restarting sampling after time limit reached")
        if (
            node.depth >= self.limits["depth"]
            or duration >= self.limits["time"]
            or node.id >= self.limits["max_id"]
            or (self.memory >= self.limits["memory"])
            or (self.diff >= self.limits["individual_memory"])
        ):
            if node.depth >= self.limits["depth"]:
                _limit_reached = "Depth"
            if duration >= self.limits["time"]:
                _limit_reached = "Time"
            if node.id >= self.limits["max_id"]:
                _limit_reached = "Max_id"
            if self.memory >= self.limits["memory"]:
                _limit_reached = "Memory"
            if self.diff >= self.limits["individual_memory"]:
                _limit_reached = "Individual Memory"
            return False
        else:
            return True

    def set_memory_checkpoint(self):
        self.memory_checkpoint = psutil.Process().memory_info().rss / (1024 * 1024)

    def reset_memory_checkpoint(self):
        self.memory_checkpoint = None

    def check_memory(self):
        """
        Check if the limits have been reached.
        """
        self.memory = psutil.Process().memory_info().rss / (1024 * 1024)
        self.diff = (
            self.memory - self.memory_checkpoint if self.memory_checkpoint is not None else 0
        )
        if self.diff >= self.limits["individual_memory"]:
            return False
        if self.memory >= self.limits["memory"]:
            return False
        return True

    def check_memory_crossover(self):
        """
        Check if the limits have been reached.
        """
        memory = psutil.Process().memory_info().rss / (1024 * 1024)
        if memory >= self.limits["memory_crossover"]:
            return False
        return True

    def check_batch_pass_time(self, node, check_memory=False):
        """
        Check if the limits have been reached.
        """
        if self.batch_shape is None:
            assert self.compile_fn is not None, (
                "Compile function must be provided if limiting batch pass time"
            )
            return True
        if check_memory:
            self.set_memory_checkpoint()
        model = self.compile_fn(node)
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
        if check_memory:
            memcheck = self.check_memory()
            if not memcheck:
                print(f"Model too large - {self.diff} MB")
                return False
        dur = 0
        timer = Timer()
        for _ in range(self.n_batch_passes):
            timer.start()
            model(torch.randn(self.batch_shape).to(self.device))
            dur += timer()
        duration = dur / self.n_batch_passes
        print(f"Batch pass duration: {duration}")
        if duration >= self.limits["batch_pass_seconds"]:
            return False
        return True

    def check_build_safe(self, node):
        param_size = node.input_params["num_params"] if "num_params" in node.input_params else 0
        if "shape" not in node.output_params or "branching_factor" not in node.output_params:
            output_size = 0
        else:
            output_size = (
                reduce(lambda x, y: x * y, node.output_params["shape"])
                * node.output_params["branching_factor"]
            )
        memory_size = param_size + output_size
        return memory_size * 4 / (1024 * 1024) < self.limits["individual_memory"]

    def __str__(self):
        repr = "Limiter(\n"
        repr += f"\t{self.limits},\n"
        repr += f"\t{self.timer}\n"
        repr += f"\t{self.memory_checkpoint:.2f} MB (baseline memory)\n"
        repr += f"\t{self.memory:.2f} MB (total memory)\n"
        repr += f"\t{self.memory - self.memory_checkpoint:.2f} MB (individual memory)\n"
        repr += ")"
        return repr
