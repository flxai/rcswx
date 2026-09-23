# Recovered coauthor baseline definitions

## Result

All four architecture definitions needed for the four table pairings are present in the latest uploaded files. ResNet34 and Mixer d12 no longer require reconstruction. The coauthor identified these files as the definitions used in their experiments. No corresponding historical Git commit or full benchmark configuration was supplied with them.

The table below reports **static operation-occurrence counts** produced by expanding the source templates and applying the tokenization in the uploaded `build_baseline`. These match the four node counts quoted in the conversation. They are not counts of PyTorch modules, learned parameters, or RCSWX alignment tokens. Runtime `serialise()` lengths have not yet been measured.

| Table model | Exact source symbol | Source assignment | Expanded operations | Historical target |
|---|---|---|---:|---:|
| ResNet18 | `resnet18_no_maxpool` | `source/baselines/resnet.py:74–99` | 267 | 267 |
| ResNet34 | `resnet34_no_maxpool` | `source/baselines/resnet.py:127–176` | 499 | 499 |
| MLP-Mixer d8 | `mlpmixer_d8` | `source/baselines/mlpmixer.py:58–86` | 264 | 264 |
| MLP-Mixer d12 | `mlpmixer_d12` | `source/baselines/mlpmixer.py:88–128` | 392 | 392 |

The source invokes residual blocks in stage counts **[2, 2, 2, 2]** for ResNet18 and **[3, 4, 6, 3]** for ResNet34, with widths 64, 128, 256 and 512. Mixer d8 and d12 contain eight and twelve explicit copies of `mlpmixer_layer`, respectively. The full nested source definitions, not these summaries, specify the fixtures.

The four primary benchmark pairs are:

1. `resnet18_no_maxpool` × `mlpmixer_d8` (267 × 264 source operations).
2. `resnet18_no_maxpool` × `mlpmixer_d12` (267 × 392).
3. `resnet34_no_maxpool` × `mlpmixer_d8` (499 × 264).
4. `resnet34_no_maxpool` × `mlpmixer_d12` (499 × 392).

## Contents

- `source/baselines/`: all six newly uploaded Python files, byte-for-byte unchanged, with their normal package filenames restored. The manifest maps these filenames to the upload names containing `(1)`.
- `definitions.py`: twelve fully expanded model-definition string constants and `MODEL_DEFINITIONS`; importable without einsearch or PyTorch. These are architecture descriptions, **not instantiated PyTorch modules**.
- `expanded/`: one expanded, otherwise unmodified definition string per model.
- `trees/`: one ordered syntax tree per model; operation names and child boundaries are retained. These are static syntax trees, **not runtime genotypes** with inferred shapes, callbacks, IDs, limiters or buffers.
- `operations/`: ordered source-operation expressions and the loader's mapped factory expressions. Neither list is represented as runtime `operation.name` values.
- `manifest.json`: source SHA-256 checksums, definition/tree/operation-stream hashes, source ranges, counts, all model names and benchmark pairings.
- `recover.py`: dependency-free, restricted-AST recovery script. It neither imports the uploaded package nor calls its `eval`-based loader.
- `test_recovery.py`: static extraction, checksum, topology and importability checks.

Run from this directory:

```sh
python recover.py
python -m unittest -v test_recovery
```

## Important source details

**Use the entire newly supplied ResNet file.** Its shared `resnet_block` shortcut is `computation[identity]` (`resnet.py:44`). The earlier uploaded version used bare `identity`. Keeping the older helper while adding only the new ResNet34 assignment would not preserve the new source definition or its operation count. Do not remove this wrapper when creating benchmark genotypes.

**The loader mismatch is resolved at the baseline-loader level.** The new `source/baselines/__init__.py:24` accepts `limits=`, and lines 113–127 use those limits to construct a limiter and iterative sampler. This does not establish compatibility with every version of the surrounding einsearch code.

**The package initializer does not re-export the two recovered symbols.** Its import lines and `baseline_dict` omit `resnet34_no_maxpool` and `mlpmixer_d12`; the definitions exist in their submodules. Use explicit submodule imports rather than assuming `from baselines import *` provides them. The copied initializer remains unchanged.

**Preserve source punctuation and ordering.** Some DSL child lists have missing or trailing commas. The uploaded loader discards delimiters while extracting operations. Recovery preserves the original strings, verifies the explicit bracket structure and child arities, and verifies agreement of the ordered tree traversal with the loader's operation stream. It does not silently rewrite the files into a stricter DSL.

**The other models are source-specific variants.** The archive also recovers:

| Symbol | Expanded source operations |
|---|---:|
| `resnet18_conv7x7_no_maxpool` | 267 |
| `mlpmixer_d2` | 72 |
| `mlpmixer_d4` | 136 |
| `vit_d2` | 108 |
| `vit_d4` | 208 |
| `vit_d8` | 408 |
| `wideresnet16_4` | 194 |
| `convnextv2_tiny` | 406 |

`vit.py` assigns `prenorm_transformer_layer` twice; the later assignment, lines 84–102, is the one used by the exported `vit_d*` strings. That overwrite is retained. The model strings use `mhsa_h4`; the separately defined `mhsa_h8` helper is not substituted.

Only **one named ConvNeXt model**, `convnextv2_tiny`, is supplied. Its source comments explicitly describe a standard rather than depthwise 7×7 convolution, ReLU in place of GELU, omission of Global Response Normalization and replacement of LayerNorm with `norm`. The actual model expression contains block-call counts [3, 3, 7, 3] by width (96, 192, 384, 768), with `computation[identity]` after the seven width-384 blocks. These are observations of the supplied expression, not a correction to it. Do not substitute a standard-library ConvNeXtV2 definition or infer extra variants from its name.

## Building runtime fixtures in the local einsearch environment

The following follows the coauthor's reported two-step route. It assumes the local agent has selected a compatible einsearch checkout and installed the supplied baseline files there **without overwriting unrelated changes**. It does not provide missing historical input dimensions or limiter values.

```python
from copy import deepcopy
from baselines import build_baseline
from baselines.resnet import resnet18_no_maxpool, resnet34_no_maxpool
from baselines.mlpmixer import mlpmixer_d8, mlpmixer_d12

# Existing experiment objects: input_params, sampler, limiter.
# Use the coauthor's actual configuration for historical reproduction.
specs = {
    "resnet18_no_maxpool": (resnet18_no_maxpool, 267),
    "resnet34_no_maxpool": (resnet34_no_maxpool, 499),
    "mlpmixer_d8": (mlpmixer_d8, 264),
    "mlpmixer_d12": (mlpmixer_d12, 392),
}
roots = {}
for name, (definition, expected_nodes) in specs.items():
    baseline_root = build_baseline(
        definition, deepcopy(input_params), limits=deepcopy(limiter.limits)
    )
    before = list(baseline_root.serialise())
    replay = [node.operation for node in before]
    # Reset the limiter's timer for this separate reconstruction.
    sampler.limiter.timer.start()
    root = sampler.sample(deepcopy(input_params), replay)
    after = list(root.serialise())
    if replay:
        raise RuntimeError(f"{name}: unconsumed operations after replay")
    if len(before) != expected_nodes or len(after) != expected_nodes:
        raise RuntimeError(f"{name}: unexpected runtime derivation-node count")
    if [node.operation.name for node in before] != [node.operation.name for node in after]:
        raise RuntimeError(f"{name}: operation replay changed the architecture")
    # Also compare parent-child structure and shape metadata in the benchmark's
    # existing genotype serializer; the checks above do not cover those fields.
    roots[name] = root

# Optional executable PyTorch backbone, outside alignment-only timings:
# model = roots["resnet34_no_maxpool"].build(roots["resnet34_no_maxpool"])
```

The example has **not** been run against einsearch here. Runtime registry availability, shape inference, limit checks and the constructed network remain validation steps for the local agent. Avoid creating all large tensor models merely to time genotype alignment. Do not count the optional task wrapper as part of a genotype unless the historical experiment did so.

## Handoff implications and remaining checks

Replace “missing” or “reconstructed” for these four fixtures with **coauthor-confirmed source supplied; static extraction verified**. Pin the uploaded file hashes. Keep the original source/tree structure, relative-width expressions, explicit identities and all wrappers. Match both source operations and runtime genotypes before reporting reference/native timing ratios.

This recovery does **not** modify `RCSWX_BENCHMARK_HANDOFF.md`, the repository or the original uploads. It supplies fixtures and recovery evidence for the local agent.

Still record the historical YAML and actual `input_params`, sampler/grammar/limiter configuration, source revision/local changes, node-count convention, exact alignment entry point, `collapse_corners` setting and timing/resource protocol. The supplied baseline files contain **no table measurement call**, so corner collapsing cannot be inferred from them.

As discussed in the conversation, the implementation exposes a boolean corner-collapse option with a fixed 0.25 offset threshold. The historical flag is **unknown**, not silently set to true in this package. Benchmark reference/native with matching flags and keep on/off configurations separate. Recheck the actual pinned algorithm sources and public interfaces before using the flag. Do not equate that threshold with a guaranteed percentage saving or assume pruned/unpruned outputs coincide.

Verified here: exact string expansion, source hashes, delimiter/arity consistency, ordered operation-stream agreement, all four source-count targets, and importability of the generated strings. **Not verified here:** PyTorch construction/forward execution, original/native alignment equality, historical distances, speed or memory results.
