import pytest
from rcswx import Limits, TensorSpec, crossover, edit_path, prepare
from rcswx.types import ParameterBinding
from torch import nn


class StatefulText(str):
    pass


class StatefulNumber(float):
    pass


@pytest.mark.parametrize("route", ["contract", "binding", "limits"])
def test_tensor_free_metadata_rejects_state_carrying_scalar_subclasses(route):
    scalar = StatefulNumber(30) if route == "limits" else StatefulText("float32")
    scalar.parent = nn.Linear(4, 4)
    with pytest.raises((TypeError, ValueError)):
        if route == "contract":
            TensorSpec((2, 4), dtype=scalar)
        elif route == "binding":
            ParameterBinding("weight", (4, 4), scalar)
        else:
            Limits(deadline_seconds=scalar)


def test_signed_zero_preserves_logical_identity_and_zero_edit_endpoints():
    spec = TensorSpec((2, 4))
    a, b = (
        prepare(nn.Sequential(nn.Dropout(zero), nn.BatchNorm1d(4, momentum=zero)), input_spec=spec)
        for zero in (-0.0, 0.0)
    )
    path = edit_path(a, b)
    assert path.cost_ticks == 0
    assert a.architecture_key == b.architecture_key
    child = crossover(a, b, path=path, seed=42)
    assert prepare(child, input_spec=spec).architecture_key == b.architecture_key
