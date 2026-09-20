"""Architecture alignment and crossover for a declared PyTorch grammar subset."""

from ._core import PROFILE_VERSION as PROFILE_VERSION
from ._core import RNG_VERSION as RNG_VERSION
from .errors import (
    AmbiguousRepresentation,
    BudgetExceeded,
    Cancelled,
    InputContractMismatch,
    InternalInvariant,
    InvalidEditSelection,
    NoAdmissibleAlignment,
    OutputContractMismatch,
    RcswxError,
    SamplingExhausted,
    StaleEditPath,
    UnsupportedArchitecture,
    UnsupportedConfiguration,
    UnsupportedEffect,
    UnsupportedOperator,
    UnsupportedSharing,
)
from .frontend import prepare
from .policies import Limits
from .types import Architecture, TensorSpec

__all__ = [
    "PROFILE_VERSION",
    "RNG_VERSION",
    "Architecture",
    "TensorSpec",
    "Limits",
    "prepare",
    "RcswxError",
    "UnsupportedArchitecture",
    "UnsupportedOperator",
    "UnsupportedEffect",
    "UnsupportedSharing",
    "UnsupportedConfiguration",
    "AmbiguousRepresentation",
    "InputContractMismatch",
    "OutputContractMismatch",
    "StaleEditPath",
    "NoAdmissibleAlignment",
    "InvalidEditSelection",
    "SamplingExhausted",
    "BudgetExceeded",
    "Cancelled",
    "InternalInvariant",
]

__version__ = "0.1.0"
