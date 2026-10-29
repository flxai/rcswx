# Reference compatibility

[README](../README.md) · [Portable guide](portable.md) · [PyTorch guide](pytorch.md)

RCSWX is a corrected port, not a bug-for-bug replica of the original `einsearch`
implementation. For the original implementation and paper experiments, see
[rcswx-paper](https://github.com/flxai/rcswx-paper).

The independent oracle is pinned to commit
`3f44ddf086bee0c213404e240ee0adf99e3e1501` and hash-checked by
[`tests/reference/original.py`](../tests/reference/original.py).

## Intentional operator corrections

The following corrections deliberately differ from that source:

- Recursive edits and wrapper boundaries use physical source occurrences rather
  than remapped trace coordinates. Swap bookkeeping distinguishes wrappers even
  when their logical IDs repeat. Removing a binary wrapper serializes its
  branches in the retained path's order, not by the recursive swap flag. An
  added wrapper closes at its own boundary edits, not at a parent-two removal
  that carries its logical ID.
- An added wrapper keeps a nested wrapper that ends its span, including a
  matched wrapper retained from parent two. The original closes the added
  wrapper before that nested wrapper, which then follows it.
- Adding a binary wrapper requires added material only in a branch that would
  otherwise be empty. The original can also require an addition in the retained
  branch, including an impossible empty enabler group.
- Zero-cost matches do not replace unchanged modules, routing functions, or
  wrapper payloads. When an accompanying binary-wrapper match requires a branch
  reorientation, application relinks its children and rebinds its anchor ID
  without copying the matched subtree from parent one. A match accompanies the
  preceding nonzero edit in application order.
- Removing the only module of a routing or single-module branching wrapper
  fails when that wrapper stays. The original's nested-wrapper dependencies
  allow such selections, then promote a routing or aggregation function into the
  module slot. Sampling is unchanged; application raises
  `rcswx.UnrealizableSelectionError`, a `ValueError`. `validated_crossover`
  counts it as a failed try and draws again, as the original did after its
  malformed child failed to build.

These corrections can change edit application order, valid selections, offspring,
and failure behavior.

## What the reference sampler means

`sampler="reference"` selects the historical NumPy/SciPy numerical route on the
**current** edit plan. It does not restore the original operator's bugs or
guarantee historical seed-to-child identity. Native and reference samplers share
the discrete probability law, not the RNG stream.

See [sampling and replay](portable.md#sampling-and-replay) for installation,
seeding, and a runnable reference-mode example.

## What the checks establish

Reference tests retain exact comparisons for unchanged behavior. The known
branch-constraint correction is explicit in the oracle comparison; separate
regressions check wrapper boundaries, ordered topology, and payload ownership.
Passing those checks is not a claim of universal original-output parity or
tensor-level validity for every generated architecture.

Selecting every edit does not guarantee parent one's topology. The remaining
known failures involve a binary wrapper whose branches the path traverses in
swapped order: an edit inside it can name the other branch's source occurrence.
The original records the same node ID but applies a remapped coordinate. Here,
application edits the wrong module, so the child can differ from parent one, or
application fails with a missing mutation target or the removal guard above.
A swapped zero-cost binary-wrapper match that precedes every nonzero edit in
application order accompanies none, so it keeps parent two's branch order even
when every edit is selected.
