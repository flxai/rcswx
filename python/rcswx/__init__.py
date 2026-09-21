"""Reference-faithful recursive constrained Smith–Waterman genotype crossover."""

from .api import distance, edit_path
from .crossover import CrossoverResult, crossover, crossover_with_report
from .genotype import DerivationTreeNode, Operation
from .limiter import Limiter
from .pcfg import PCFG, OutOfOptionsError
from .pipeline import GenerationResult, gated_crossover, generation_step, validated_crossover
from .reconstruction import Reconstructor
from .recursive import Alignment, MatrixOperation, raw_crossover
from .sampling import select_operations, valid_combinations

__all__ = [
    "Alignment",
    "CrossoverResult",
    "DerivationTreeNode",
    "GenerationResult",
    "Limiter",
    "MatrixOperation",
    "Operation",
    "OutOfOptionsError",
    "PCFG",
    "Reconstructor",
    "crossover",
    "crossover_with_report",
    "distance",
    "edit_path",
    "gated_crossover",
    "generation_step",
    "raw_crossover",
    "select_operations",
    "validated_crossover",
    "valid_combinations",
]

__version__ = "0.1.0"
