import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from rcswx import Limits, SamplingExhausted, TensorSpec, crossover_with_report, edit_path, prepare
from torch import nn

_recipe = st.lists(
    st.tuples(
        st.sampled_from(("identity", "relu", "linear", "dropout", "normalization")),
        st.integers(1, 8),
    ),
    min_size=1,
    max_size=8,
)


def network(recipe):
    layers, width = [], 4
    for kind, value in recipe:
        if kind == "identity":
            layers.append(nn.Identity())
        elif kind == "relu":
            layers.append(nn.ReLU())
        elif kind == "linear":
            layers.append(nn.Linear(width, value, bias=value % 2 == 0))
            width = value
        elif kind == "dropout":
            layers.append(nn.Dropout(value / 10))
        else:
            layers.append(
                nn.BatchNorm1d(width, affine=value % 2 == 0, track_running_stats=value % 3 == 0)
            )
    layers.append(nn.Linear(width, 4))
    return nn.Sequential(*layers).eval()


@given(_recipe, _recipe, st.integers(0, (1 << 64) - 1))
@settings(max_examples=40, deadline=None, derandomize=True, database=None)
def test_generated_supported_children_preserve_selected_descriptors_and_training(left, right, seed):
    source, target = network(left), network(right)
    spec = TensorSpec((2, 4))
    path = edit_path(source, target, input_spec=spec)
    rng = torch.get_rng_state()
    try:
        result = crossover_with_report(
            source, target, path=path, seed=seed, inherit=True, limits=Limits(max_attempts=4)
        )
    except SamplingExhausted as error:
        assert error.report["attempts"] == len(error.report["rejections"]) == 4
        for rejection in error.report["rejections"]:
            rejected, _ = path._project(rejection["mask"])
            final_width = 4
            for operation in rejected:
                if operation.kind == "linear":
                    final_width = operation.get("out_features")
            assert final_width != 4
        assert torch.equal(torch.get_rng_state(), rng)
        return
    assert torch.equal(torch.get_rng_state(), rng)
    operations, _ = path._project(result.report["selected_mask"])
    child = result.model
    assert prepare(child, input_spec=spec).operations == operations
    output = child(torch.ones(2, 4))
    assert tuple(output.shape) == spec.shape
    output.square().sum().backward()
    assert all(p.grad is None for parent in (source, target) for p in parent.parameters())
    assert all(p.grad is not None for p in child.parameters())
    donor_pointers = {
        t.data_ptr()
        for parent in (source, target)
        for t in (*parent.parameters(), *parent.buffers())
    }
    assert all(t.data_ptr() not in donor_pointers for t in (*child.parameters(), *child.buffers()))
