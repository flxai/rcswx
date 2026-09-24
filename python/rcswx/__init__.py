"""Framework-free RCSWX structural API with lazy legacy integrations."""

from __future__ import annotations

from importlib import import_module

from .api import Alignment, Selection, apply_edits, distance, edit_path
from .crossover import CrossoverResult, crossover, crossover_with_report
from .portable import Architecture, from_dict, from_json, to_dict, to_json
from .recursive import MatrixOperation, raw_crossover

__version__ = "0.1.0"

_LAZY_EXPORTS = {
    "DerivationTreeNode": (".genotype", "DerivationTreeNode"),
    "Operation": (".genotype", "Operation"),
    "Limiter": (".limiter", "Limiter"),
    "PCFG": (".pcfg", "PCFG"),
    "OutOfOptionsError": (".pcfg", "OutOfOptionsError"),
    "Reconstructor": (".reconstruction", "Reconstructor"),
    "select_operations": (".sampling", "select_operations"),
    "valid_combinations": (".sampling", "valid_combinations"),
    "NativeRng": ("._core", "NativeRng"),
    "validated_crossover": (".pipeline", "validated_crossover"),
    "gated_crossover": (".pipeline", "gated_crossover"),
    "generation_step": (".pipeline", "generation_step"),
    "GenerationResult": (".pipeline", "GenerationResult"),
}


def __getattr__(name: str):
    try:
        module_name, attribute = _LAZY_EXPORTS[name]
    except KeyError as error:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from error
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted([*globals(), *_LAZY_EXPORTS])


__all__ = [
    "Alignment",
    "Architecture",
    "CrossoverResult",
    "MatrixOperation",
    "NativeRng",
    "Selection",
    "apply_edits",
    "crossover",
    "crossover_with_report",
    "distance",
    "edit_path",
    "from_dict",
    "from_json",
    "to_dict",
    "to_json",
    "raw_crossover",
]
