"""Reference subset law, with owned native dependency enumeration.

The SciPy grid and NumPy choice intentionally remain in the caller's numerical
runtime. Replacing either with a private generator or another normalization
changes seeded outcomes or ambient RNG transitions.
"""

import numpy as np
from scipy.stats import skewnorm

from . import _core


def valid_combinations(operations):
    """Return lexicographically ordered masks and source-ordered subset costs."""
    identities = {}

    def identity(operation):
        return identities.setdefault(operation.id, len(identities))

    def groups(dependencies):
        return [
            [identity(operation) for operation in group]
            for group in dependencies
            if type(group) is list
        ] + [[identity(operation) for operation in dependencies if type(operation) is not list]]

    records = [
        (
            identity(operation),
            operation.value,
            groups(operation.enabler_ops),
            groups(operation.disabler_ops),
        )
        for operation in operations
    ]
    return _core.valid_combinations(records)


def select_operations(operations, skewness=0):
    masks, values = valid_combinations(operations)
    distribution = skewnorm(skewness)
    sample_at = np.linspace(distribution.ppf(0.01), distribution.ppf(0.99), int(values[-1] * 4))
    samples = distribution.pdf(sample_at)
    probabilities = [samples[int(value * 4) - 1] for value in values]
    probabilities /= np.sum(probabilities)
    selected = np.random.choice(masks, p=probabilities)
    return [operation for index, operation in enumerate(operations) if selected[index] == "1"]
