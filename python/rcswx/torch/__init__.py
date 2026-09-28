"""Optional PyTorch capture, build, and fresh-module crossover adapter."""

from importlib import import_module

try:
    import_module("torch")
except ModuleNotFoundError as error:
    if error.name == "torch":
        raise ModuleNotFoundError(
            "rcswx.torch requires the optional PyTorch dependency; install rcswx[torch]"
        ) from error
    raise

from .build import build
from .converters import linear_relu_sequential_converter
from .crossover import crossover, crossover_imported, crossover_with_report
from .graph_import import import_model
from .provenance import capture, captured_from_manifest, load, save
from .registry import (
    ImporterRegistration,
    register_importer,
    registered_importers,
    unregister_importer,
)
from .types import (
    BuildOptions,
    CapturedArchitecture,
    ConverterResult,
    ModuleCrossoverResult,
    StaleProvenanceError,
    TorchAdapterError,
    UnsafeManifestError,
    UnsupportedModuleError,
)

__all__ = [
    "BuildOptions",
    "CapturedArchitecture",
    "ConverterResult",
    "ImporterRegistration",
    "ModuleCrossoverResult",
    "StaleProvenanceError",
    "TorchAdapterError",
    "UnsafeManifestError",
    "UnsupportedModuleError",
    "build",
    "capture",
    "captured_from_manifest",
    "crossover",
    "crossover_imported",
    "crossover_with_report",
    "import_model",
    "linear_relu_sequential_converter",
    "load",
    "register_importer",
    "registered_importers",
    "save",
    "unregister_importer",
]
