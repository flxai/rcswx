"""Optional historical SciPy/NumPy sampling policy.

This module is imported only for ``sampler='reference'``. A minimal native
installation can import ``rcswx`` and ``rcswx.sampling`` without scientific
Python; an explicit import of this policy requires its declared extra.
"""

try:
    import numpy as np
    from scipy.stats import skewnorm
except ModuleNotFoundError as error:
    raise ModuleNotFoundError(
        "sampler='reference' requires the optional rcswx[reference] extra (NumPy and SciPy)"
    ) from error


def probabilities(values, skewness=0):
    """Reproduce the pinned SciPy probability construction exactly."""
    distribution = skewnorm(skewness)
    sample_at = np.linspace(
        distribution.ppf(0.01),
        distribution.ppf(0.99),
        int(values[-1] * 4),
    )
    samples = distribution.pdf(sample_at)
    result = [samples[int(value * 4) - 1] for value in values]
    result /= np.sum(result)
    return result


def select_mask(masks, costs, skewness=0):
    """Draw one already-enumerated mask from the historical global stream."""
    return np.random.choice(masks, p=probabilities(costs, skewness))


def select_operations(operations, skewness=0):
    """Use the historical ambient NumPy draw after native enumeration."""
    from .sampling import valid_combinations

    masks, values = valid_combinations(operations)
    selected = select_mask(masks, values, skewness)
    return [operation for index, operation in enumerate(operations) if selected[index] == "1"]
