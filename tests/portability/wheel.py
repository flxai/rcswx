#!/usr/bin/env python3
"""Install one release wheel into isolated environments for every support surface.

This is deliberately a standard-library executable rather than a pytest test: it
must create clean environments and prove that the selected wheel, rather than
this checkout or an editable install, supplies each public surface.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
import zipfile
from email.parser import BytesParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PORTABLE_EXAMPLE = ROOT / "examples" / "portable.py"
PYTORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"
PYPI_INDEX = "https://pypi.org/simple"
TORCH_EXAMPLES = (
    "parents.py",
    "distance.py",
    "edit_path.py",
    "crossover.py",
    "crossover_report.py",
    "validated_crossover.py",
    "torch_module_crossover.py",
    "torch_converter.py",
)
SURFACES = {
    "minimal": (),
    "torch": ("torch",),
    "reference": ("reference",),
    "torch-reference": ("torch", "reference"),
}


def run(command: list[str], *, cwd: Path) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def python_in(environment: Path) -> Path:
    return environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def wheel_metadata(wheel: Path):
    with zipfile.ZipFile(wheel) as archive:
        names = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
        if len(names) != 1:
            raise AssertionError(f"expected one METADATA file in {wheel}, found {names!r}")
        return BytesParser().parsebytes(archive.read(names[0]))


def require_marker(requirement: str) -> str:
    match = re.match(r"([A-Za-z0-9_.-]+)", requirement)
    if match is None:
        raise AssertionError(f"could not parse wheel requirement {requirement!r}")
    return match.group(1).lower().replace("_", "-")


def assert_metadata(wheel: Path) -> None:
    metadata = wheel_metadata(wheel)
    requires_python = metadata["Requires-Python"]
    if requires_python is None or {
        constraint.strip() for constraint in requires_python.split(",")
    } != {">=3.12", "<3.15"}:
        raise AssertionError(f"unexpected Requires-Python: {requires_python!r}")
    extras = set(metadata.get_all("Provides-Extra") or [])
    if extras != {"torch", "reference"}:
        raise AssertionError(f"unexpected extras: {sorted(extras)!r}")

    requirements = metadata.get_all("Requires-Dist") or []
    ungated = [requirement for requirement in requirements if ";" not in requirement]
    if ungated:
        raise AssertionError(f"minimal wheel has mandatory dependencies: {ungated!r}")

    by_extra = {"torch": set(), "reference": set()}
    for requirement in requirements:
        normalized = require_marker(requirement)
        for extra in by_extra:
            if re.search(rf"extra\s*==\s*['\"]{extra}['\"]", requirement):
                by_extra[extra].add(normalized)
    if by_extra["torch"] != {"torch", "psutil", "rich", "termcolor", "tqdm"}:
        raise AssertionError(f"unexpected torch requirements: {sorted(by_extra['torch'])!r}")
    if by_extra["reference"] != {"numpy", "scipy"}:
        raise AssertionError(
            f"unexpected reference requirements: {sorted(by_extra['reference'])!r}"
        )


def run_python(environment: Path, source: str, *, cwd: Path) -> None:
    run([str(python_in(environment)), "-I", "-c", source], cwd=cwd)


def assert_installed_root(environment: Path, available: tuple[str, ...], parallel: bool) -> None:
    run_python(
        environment,
        f"""
import importlib.machinery
import importlib.util
import pathlib
import sys
import rcswx
from rcswx import _core
location = pathlib.Path(rcswx.__file__).resolve()
extension = pathlib.Path(_core.__file__).resolve()
prefix = pathlib.Path(sys.prefix).resolve()
assert location.is_relative_to(prefix), (location, prefix)
assert extension.is_relative_to(prefix), (extension, prefix)
assert any(extension.name.endswith(suffix) for suffix in importlib.machinery.EXTENSION_SUFFIXES), extension
assert _core.PARALLEL_CAPABLE is {parallel!r}
def parent(changed):
    items = [
        ("computation", (f"linear({{16 + 2 * index + int(changed and index == 24)}})",))
        for index in range(48)
    ]
    tree = items[-1]
    for leaf in reversed(items[:-1]):
        tree = ("routing", ("identity",), ("sequential", leaf, tree), ("identity",))
    return rcswx.Architecture.from_tree(tree)
pair = parent(False), parent(True)
serial = rcswx.edit_path(*pair, workers=1)
assert serial.execution["pool_capacity"] is None
if _core.PARALLEL_CAPABLE:
    native = rcswx.edit_path(*pair, workers=2)
    assert native._native.paths_json() == serial._native.paths_json()
    assert native.stats == serial.stats
    assert native.execution["worker_limit"] <= 2
    if native.execution["worker_limit"] == 2:
        assert native.execution["parallel_cells"] > 0
else:
    try:
        rcswx.edit_path(*pair, workers=2)
    except RuntimeError:
        pass
    else:
        raise AssertionError("serial-only wheel accepted workers=2")
available = {available!r}
for name in ("torch", "numpy", "scipy", "psutil", "rich", "termcolor", "tqdm"):
    loaded = name in sys.modules or any(module.startswith(name + ".") for module in sys.modules)
    assert not loaded, (name, sorted(sys.modules))
    if name in available:
        assert importlib.util.find_spec(name) is not None, name
    else:
        assert importlib.util.find_spec(name) is None, name
""",
        cwd=environment.parent,
    )


def assert_missing_extra(environment: Path, module: str, extra: str) -> None:
    run_python(
        environment,
        f"""
import importlib
try:
    importlib.import_module({module!r})
except ModuleNotFoundError as error:
    assert "rcswx[{extra}]" in str(error), str(error)
else:
    raise AssertionError({module!r} + " unexpectedly imported without rcswx[{extra}]")
""",
        cwd=environment.parent,
    )


def install_surface(
    root: Path, wheel: Path, name: str, extras: tuple[str, ...], interpreter: str
) -> Path:
    environment = root / name
    run(["uv", "venv", "--python", interpreter, str(environment)], cwd=root)
    if extras:
        requirement = f"rcswx[{','.join(extras)}] @ {wheel.as_uri()}"
        run(
            [
                "uv",
                "pip",
                "install",
                "--index",
                PYTORCH_CPU_INDEX,
                "--index",
                PYPI_INDEX,
                "--python",
                str(python_in(environment)),
                requirement,
            ],
            cwd=root,
        )
    else:
        run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                str(python_in(environment)),
                "--no-deps",
                str(wheel),
            ],
            cwd=root,
        )
    return environment


def run_example(environment: Path, name: str, root: Path) -> None:
    # Keep the installed package isolated, but allow sibling example helpers.
    run_python(
        environment,
        "import runpy, sys, rcswx\n"
        f"sys.path.insert(0, {str(ROOT / 'examples')!r})\n"
        f"runpy.run_path({str(ROOT / 'examples' / name)!r}, run_name='__main__')",
        cwd=root,
    )


def check(wheel: Path, interpreter: str, parallel: bool) -> None:
    assert_metadata(wheel)
    with tempfile.TemporaryDirectory(prefix="rcswx-wheel-") as temporary:
        root = Path(temporary)
        environments = {
            name: install_surface(root, wheel, name, extras, interpreter)
            for name, extras in SURFACES.items()
        }

        assert_installed_root(environments["minimal"], (), parallel)
        assert_missing_extra(environments["minimal"], "rcswx.torch", "torch")
        assert_missing_extra(environments["minimal"], "rcswx.sampling_reference", "reference")
        run_example(environments["minimal"], PORTABLE_EXAMPLE.name, root)

        assert_installed_root(
            environments["torch"], ("torch", "psutil", "rich", "termcolor", "tqdm"), parallel
        )
        run_python(environments["torch"], "import rcswx.torch", cwd=root)
        for example in TORCH_EXAMPLES:
            run_example(environments["torch"], example, root)

        assert_installed_root(environments["reference"], ("numpy", "scipy"), parallel)
        run_python(environments["reference"], "import rcswx.sampling_reference", cwd=root)
        run_example(environments["reference"], "reference_sampling.py", root)

        assert_installed_root(
            environments["torch-reference"],
            ("torch", "numpy", "scipy", "psutil", "rich", "termcolor", "tqdm"),
            parallel,
        )
        run_python(
            environments["torch-reference"],
            "import rcswx.sampling_reference; import rcswx.torch",
            cwd=root,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", required=True, type=Path)
    parser.add_argument("--expect-parallel", action="store_true")
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="interpreter compatible with the selected wheel (default: this interpreter)",
    )
    arguments = parser.parse_args()
    wheel = arguments.wheel.resolve()
    if not wheel.is_file() or wheel.suffix != ".whl":
        parser.error(f"--wheel must name one wheel file, got {wheel}")
    check(wheel, arguments.python, arguments.expect_parallel)


if __name__ == "__main__":
    main()
