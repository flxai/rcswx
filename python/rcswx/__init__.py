"""Architecture alignment and crossover for a declared PyTorch grammar subset."""

from ._core import PROFILE_VERSION as PROFILE_VERSION
from ._core import RNG_VERSION as RNG_VERSION
from .api import CancellationToken, Edit, EditPath, distance, edit_path
from .crossover import CrossoverResult, crossover, crossover_with_report
from .errors import (
    AmbiguousRepresentation,
    BudgetExceeded,
    Cancelled,
    InputContractMismatch,
    InternalInvariant,
    InvalidEditSelection,
    MissingDonor,
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
    "CancellationToken",
    "Edit",
    "EditPath",
    "distance",
    "edit_path",
    "crossover",
    "crossover_with_report",
    "CrossoverResult",
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
    "MissingDonor",
    "NoAdmissibleAlignment",
    "InvalidEditSelection",
    "SamplingExhausted",
    "BudgetExceeded",
    "Cancelled",
    "InternalInvariant",
]

__version__ = "0.1.0"
