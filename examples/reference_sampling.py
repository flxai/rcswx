"""Reference-mode selection needs rcswx[reference], but not PyTorch."""

import numpy as np
from rcswx import Architecture, apply_edits, edit_path

first = Architecture.from_tree(("computation", ("relu",)))
second = Architecture.from_tree(("computation", ("sigmoid",)))
plan = edit_path(first, second)

# This mode deliberately follows NumPy's global stream, not an RCSWX seed.
np.random.seed(0)
selection = plan.sample(sampler="reference")
child = apply_edits(plan, selection)
print(f"distance={plan.distance}, selected_cost={selection.cost}")
print(child.to_json())
