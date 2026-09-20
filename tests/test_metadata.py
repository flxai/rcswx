import pytest
from rcswx import Limits, TensorSpec
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
