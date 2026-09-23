# rcswx

Recursive Constrained Smith–Waterman crossover for grammar-based neural architecture search: edit distances, edit paths, and offspring generation with a Rust-backed Python API.

Reimplementation of [Evolutionary Architecture Search through Grammar-Based Sequence Alignment](https://arxiv.org/abs/2512.04992). The paper's original implementation is [flxai/rcswx-paper](https://github.com/flxai/rcswx-paper).

## Usage

Inputs are grammar derivation trees describing a network, not arbitrary PyTorch or TensorFlow models. Bundled [grammars](https://github.com/flxai/rcswx/tree/main/python/rcswx/grammars) include einspace and HNASBench201 with PyTorch model construction.

- [`distance(a, b)`](https://github.com/flxai/rcswx/blob/main/python/rcswx/api.py) returns the edit distance.
- [`edit_path(a, b)`](https://github.com/flxai/rcswx/blob/main/python/rcswx/api.py) returns an alignment with its edit operations.
- [`crossover(a, b)`](https://github.com/flxai/rcswx/blob/main/python/rcswx/crossover.py) returns an offspring tree.

See [examples/](https://github.com/flxai/rcswx/tree/main/examples) for parent construction, distances, edit paths, crossover, and model building.

Calls may change parents or return a parent directly. Copy parents first to keep the originals unchanged. Crossover does not transfer weights; model construction is separate.

## Installation

Requires Python 3.12–3.14. After publication on PyPI, choose one:

```sh
uv add rcswx                  # Add to a uv-managed project.
uv pip install rcswx          # Install into a virtual environment.
python -m pip install rcswx   # Alternatively, use pip in an active virtual environment.
```

Source builds require Rust/Cargo. See [source installation](https://github.com/flxai/rcswx/blob/main/examples/README.md#from-source) and [Nix Flakes](https://github.com/flxai/rcswx/blob/main/examples/README.md#nix-flakes).

## Cite

Please cite the paper if this software helps your research:

```bibtex
@misc{rcswx2025,
  author        = {Gómez Martín, Adri and Möller, Felix and
                   McDonagh, Steven and Abella, Monica and Desco, Manuel and
                   Crowley, Elliot J. and Klein, Aaron and Ericsson, Linus},
  title         = {{Evolutionary Architecture Search through Grammar-Based Sequence Alignment}},
  year          = {2025},
  eprint        = {2512.04992},
  archivePrefix = {arXiv},
  primaryClass  = {cs.NE},
  doi           = {10.48550/arXiv.2512.04992},
  url           = {https://arxiv.org/abs/2512.04992}
}
```

## License

[MIT](https://github.com/flxai/rcswx/blob/main/LICENSE.einsearch)
