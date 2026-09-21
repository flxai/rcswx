"""Reference crossover validation and outer generation recovery boundaries.

Parent selection, mutation, task heads, evaluation, and population storage remain
caller-owned. In particular, regeneration is a callback, not repeated selection
from one cached alignment or silent shape-conditioning of the subset sampler.
"""

import pickle
import random
from copy import deepcopy
from dataclasses import dataclass
from os import makedirs
from os.path import join

import torch

from .recursive import raw_crossover


def validated_crossover(
    parent1, parent2, *, rebuild, batch_shape, skewness=0, max_tries=100, limiter=None
):
    """Reconstruct and run a temporary model; return genotype and reference metadata.

    ``rebuild`` is an operation-driven reconstructor such as ``Reconstructor.re_id``.
    The raw call deliberately sits outside the build/forward retry handler.
    """
    if limiter is None:
        limiter = parent1.limiter
    root_input_params = deepcopy(parent1.input_params)
    tries = 0
    while True:
        if tries > max_tries:
            raise RuntimeError("Crossover failed to generate valid children.")
        tries += 1
        child, selected, operations, distance1, distance2, between = raw_crossover(
            parent1, parent2, skewness=skewness, limiter=limiter
        )
        try:
            child.input_params = root_input_params
            child = rebuild(child)
            model = child.build(child)
            model(torch.randn(*batch_shape))
            return child, {
                "crossover": True,
                "crossover_operations": selected,
                "crossover_distance_to_parent1": distance1,
                "crossover_distance_to_parent2": distance2,
                "crossover_distance_between_parents": between,
                "crossover_all_operations": operations,
                "crossover_skewness": skewness,
            }
        except Exception as error:
            print(error)


def gated_crossover(parent1, parent2, *, crossover_rate, rebuild, batch_shape, limiter=None):
    """Consume the caller's Python RNG at the original crossover gate."""
    if random.random() < crossover_rate:
        return validated_crossover(
            parent1, parent2, rebuild=rebuild, batch_shape=batch_shape, limiter=limiter
        )
    return parent1, {"crossover": False}


@dataclass(frozen=True, slots=True)
class GenerationResult:
    root: object
    ancestry: object
    reward: object
    sample_duration: float
    evaluation_duration: float
    attempts: int


def generation_step(
    generate, evaluate, *, limiter, n_tries=None, parents=None, results_path=None, iteration=0
):
    """Generate, validate, and evaluate one accepted individual.

    ``generate()`` returns ``(root, ancestry)`` and may select a fresh pair each time.
    ``parents()``, if supplied, returns the currently selected individual objects
    (with ``id`` and ``root``), enabling the reference debug pickle behavior. The
    caller adds the returned successful result to its population. Evaluation can
    construct its task-specific Network; raw crossover does not invent a task head.
    """
    attempts = 0
    while True:
        attempts += 1
        try:
            limiter.timer.start()
            root, ancestry = generate()
            sample_duration = limiter.timer()
            if not limiter.check_batch_pass_time(root, check_memory=True):
                print("Batch pass time or memory exceeded, trying again")
                continue
            limiter.timer.start()
            reward = evaluate(root)
            evaluation_duration = limiter.timer()
        except (RuntimeError, MemoryError) as error:
            print(f"Error in generating new individual: {error}")
            if n_tries is not None and attempts > n_tries:
                raise
        except IndexError as error:
            print(f"Error in generating new individual: {error}")
            parent1, parent2 = parents() if parents is not None else (None, None)
            if parent1 and parent2:
                # Match Evolution.step's path construction and default pickle protocol.
                debug_path = str(results_path).rsplit("/", 1)[0] + "/debug"
                makedirs(debug_path, exist_ok=True)
                for side, parent in ((1, parent1), (2, parent2)):
                    with open(
                        join(debug_path, f"parent{side}_iter{iteration}_id{parent.id}.pkl"), "wb"
                    ) as output:
                        pickle.dump(parent.root, output)
            if n_tries is not None and attempts > n_tries:
                raise
        else:
            return GenerationResult(
                root, ancestry, reward, sample_duration, evaluation_duration, attempts
            )
