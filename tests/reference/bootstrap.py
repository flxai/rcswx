"""Prepare the CPU reference runtime; does not build or install the library under test."""

import argparse
import subprocess
from pathlib import Path

REQUIREMENTS = (
    "numpy==1.26.4",
    "scipy==1.13.0",
    "torch==2.3.1+cpu",
    "psutil==5.9.8",
    "rich==13.7.1",
    "tqdm==4.66.4",
    "termcolor==2.4.0",
    "pytest==8.3.5",
    "maturin>=1.9,<2",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("environment", type=Path)
    parser.add_argument("--python", default="python3.12")
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument(
        "--benchmarks",
        action="store_true",
        help="Also install the native-inclusive memory profiler",
    )
    args = parser.parse_args()
    interpreter = args.environment / "bin" / "python"
    print(f"Reference environment preparation log: {args.log}", flush=True)
    with args.log.open("w") as log:
        if not interpreter.exists():
            subprocess.run(
                ["uv", "venv", "--python", args.python, str(args.environment)],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        subprocess.run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                str(interpreter),
                "--extra-index-url",
                "https://download.pytorch.org/whl/cpu",
                "--index-strategy",
                "unsafe-best-match",
                *REQUIREMENTS,
                *(("memray==1.20.0",) if args.benchmarks else ()),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    print(f"Reference environment prepared: {interpreter}")


if __name__ == "__main__":
    main()
