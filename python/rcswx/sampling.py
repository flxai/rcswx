"""Native-default selection with an explicit historical reference route."""

import os

from . import _core

_REFERENCE = object()


def _records(operations):
    """Marshal legacy operations without changing dependency grouping semantics."""
    identities = {}

    def identity(operation):
        return identities.setdefault(operation.id, len(identities))

    def groups(dependencies):
        # This intentionally mirrors the pinned selector: explicit branch lists
        # precede one synthetic branch containing scalar dependencies. Empty lists,
        # repeated operations, and malformed branch counts remain observable.
        return [
            [identity(operation) for operation in group]
            for group in dependencies
            if type(group) is list
        ] + [[identity(operation) for operation in dependencies if type(operation) is not list]]

    return [
        (
            identity(operation),
            operation.value,
            groups(operation.enabler_ops),
            groups(operation.disabler_ops),
        )
        for operation in operations
    ]


def valid_combinations(operations):
    """Return lexicographically ordered masks and source-ordered subset costs."""
    return _core.valid_combinations(_records(operations))


def resolve_rng(sampler, seed=None, rng=None):
    """Resolve one sampler stream for one public call.

    The native route never consults NumPy, Torch, or a Python global RNG. Entropy
    is acquired here, at the host boundary, only when the caller supplies neither
    a seed nor a stateful native stream.
    """
    if sampler == "reference":
        if seed is not None or rng is not None:
            raise ValueError("seed and rng are supported only by sampler='native'")
        return _REFERENCE
    if sampler != "native":
        raise ValueError("sampler must be 'native' or 'reference'")
    if seed is not None and rng is not None:
        raise ValueError("seed and rng are mutually exclusive")
    if rng is not None:
        if not isinstance(rng, _core.NativeRng):
            raise TypeError("rng must be an rcswx._core.NativeRng")
        return rng
    if seed is None:
        return _core.NativeRng(os.urandom(32))
    return _core.NativeRng(seed)


def probabilities(values, skewness=0, *, sampler="native"):
    """Return proposal probabilities under the selected numerical policy."""
    if sampler == "native":
        return _core.native_probabilities(values, skewness)
    if sampler == "reference":
        resolve_rng(sampler)
        from .sampling_reference import probabilities as reference_probabilities

        return reference_probabilities(values, skewness)
    raise ValueError("sampler must be 'native' or 'reference'")


def select_operations(operations, skewness=0, *, sampler="native", seed=None, rng=None):
    """Select operations with native ChaCha12 sampling by default.

    ``sampler='reference'`` is the only route which imports NumPy/SciPy or
    consumes their global stream. Native integer seeds are little-endian values
    in ``[0, 2**256)``; byte seeds must contain exactly 32 bytes.
    """
    resolved_rng = resolve_rng(sampler, seed, rng)
    if sampler == "reference":
        # Reference mode retains the historical SciPy/NumPy law and its ambient
        # NumPy stream. It is deliberately imported only on this explicit route.
        from .sampling_reference import select_operations as select_reference

        return select_reference(operations, skewness)
    mask, _cost = _core.native_select(_records(operations), skewness, resolved_rng)
    return [operation for index, operation in enumerate(operations) if mask[index] == "1"]
