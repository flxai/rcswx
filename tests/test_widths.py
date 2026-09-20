import torch
from rcswx import Limits, TensorSpec, edit_path, prepare
from rcswx.materialize import materialize
from rcswx.operators import infer
from torch import nn


def test_all_512_inferred_width_masks_preserve_endpoints_and_whole_parameters():
    def network(width):
        dimensions = [4] + [width] * 9 + [4]
        return nn.Sequential(*(nn.Linear(a, b) for a, b in zip(dimensions, dimensions[1:])))

    source, target = network(8).eval(), network(16)
    with torch.no_grad():
        for parent, offset in ((source, 1), (target, 11)):
            for index, layer in enumerate(parent):
                layer.weight.fill_((index + offset) / 10)
                layer.bias.fill_((index + offset) / 100)
    snapshots = tuple(p.clone() for model in (source, target) for p in model.parameters())
    donor_ptrs = {p.data_ptr() for model in (source, target) for p in model.parameters()}
    spec, limits = TensorSpec((2, 4)), Limits()
    a, b = (prepare(parent, input_spec=spec) for parent in (source, target))
    path = edit_path(a, b)
    assert len(path.edits) == path.cost_ticks == 9
    keys = set()
    for mask in range(512):
        operations, origins = path._project(mask)
        inference = infer(operations, spec, limits=limits)
        child, decisions = materialize(
            operations,
            spec,
            inference=inference,
            training=a.training,
            init="zeros",
            initialization_seed=0,
            inherit=True,
            donors=(source, target),
            donor_architectures=(a, b),
            provenance=origins,
            limits=limits,
        )
        widths = [16 if mask & (1 << bit) else 8 for bit in range(9)] + [4]
        assert [layer.out_features for layer in child] == widths
        assert [layer.in_features for layer in child] == [4] + widths[:-1]
        assert child(torch.ones(2, 4)).shape == (2, 4)
        for index, layer in enumerate(child):
            side, address = origins[index]
            donor = (source, target)[side][address]
            compatible = layer.weight.shape == donor.weight.shape
            assert torch.equal(
                layer.weight, donor.weight if compatible else torch.zeros_like(layer.weight)
            )
            assert torch.equal(layer.bias, donor.bias)
            assert decisions[2 * index]["action"] == ("copied" if compatible else "initialized")
        assert all(p.data_ptr() not in donor_ptrs for p in child.parameters())
        key = prepare(child, input_spec=spec).architecture_key
        keys.add(key)
        if mask == 0:
            assert key == a.architecture_key
            assert torch.equal(child(torch.ones(2, 4)), source(torch.ones(2, 4)))
        elif mask == 511:
            assert key == b.architecture_key
            assert origins[-1] == (0, 9)
    assert len(keys) == 512
    assert len(keys - {a.architecture_key, b.architecture_key}) == 510
    assert all(
        torch.equal(before, after)
        for before, after in zip(
            snapshots, (p for model in (source, target) for p in model.parameters()), strict=True
        )
    )
