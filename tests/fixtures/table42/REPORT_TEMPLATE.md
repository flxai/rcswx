# RCSWX Table 4.2 — execution report

**Status:** template; no benchmark results populated. Replace placeholders with observed values or explicit status codes, not estimates.

## Run identity

Record native/reference commits and dirty-diff hashes; release-extension path/SHA-256; environment/lockfile manifests; CPU/RAM/thread/affinity settings; fixture-manifest hash; resolved runtime configuration; execution dates and campaign budgets. State whether the reference is repaired-current, historically verified, or an explicitly labeled candidate.

## 1. Fixture audit

| Model | Symbol | Static source count | Reference runtime nodes | Native runtime nodes | Alignment tokens (reference/native) | Model/forward status |
|---|---|---:|---|---|---|---|
| ResNet18 | `resnet18_no_maxpool` | 267 | — | — | — | NOT RUN |
| ResNet34 | `resnet34_no_maxpool` | 499 | — | — | — | NOT RUN |
| Mixer d8 | `mlpmixer_d8` | 264 | — | — | — | NOT RUN |
| Mixer d12 | `mlpmixer_d12` | 392 | — | — | — | NOT RUN |

Include the source-to-factory-to-runtime mapping, exact topology/metadata comparison and fallback-free replay checks. Distinguish static counts, runtime counts, token counts and parameter/module counts.

## 2. Printed historical targets — preserve verbatim

| Pair | Printed nodes | Printed compute time | Printed distance |
|---|---:|---:|---:|
| ResNet18 × Mixer d8 | 267 / 264 | 39.054 seconds | 60.38 |
| ResNet18 × Mixer d12 | 267 / 392 | 342.383 seconds | 95.38 |
| ResNet34 × Mixer d8 | 499 / 264 | 86.633 seconds | 119.62 |
| ResNet34 × Mixer d12 | 499 / 392 | 741.57 minutes | 115.62 |

The last unit has not been independently confirmed; its literal conversion is 44,494.2 seconds. Historical algorithm/configuration/pruning/timing evidence: [state what was recovered and what remains unknown].

## 3a. Paired current reference — corner collapse off

Configuration/seed/limits: [resolved manifest]. Boundary: complete alignment-object return; no setup or fingerprinting in call time. State scalar, ordered-history, edit/dependency, parent-effect and RNG verification separately.

| Pair | Reference distance | Rust distance | Verification / status | Ref. median s | Rust median s | Eligible speedup | Matched repeats |
|---|---|---|---|---|---|---|---|
| ResNet18 × Mixer d8 | — | — | NOT RUN | — | — | — | 0 |
| ResNet18 × Mixer d12 | — | — | NOT RUN | — | — | — | 0 |
| ResNet34 × Mixer d8 | — | — | NOT RUN | — | — | — | 0 |
| ResNet34 × Mixer d12 | — | — | NOT RUN | — | — | — | 0 |

| Pair | Reference baseline / peak MiB | Rust baseline / peak MiB | Reference peak delta MiB | Rust peak delta MiB | Retained RSS MiB (ref./Rust) | Status |
|---|---|---|---|---|---|---|
| ResNet18 × Mixer d8 | — | — | — | — | — | NOT RUN |
| ResNet18 × Mixer d12 | — | — | — | — | — | NOT RUN |
| ResNet34 × Mixer d8 | — | — | — | — | — | NOT RUN |
| ResNet34 × Mixer d12 | — | — | — | — | — | NOT RUN |

List raw repetitions, min/max and completion/exclusion counts separately. State RSS-reset status and any lifetime-only fallback labels. Do not compute a ratio from missing, mismatched, failed or censored values.

## 3b. Paired current reference — corner collapse on

Configuration/seed/limits: [resolved manifest]. Boundary: complete alignment-object return; no setup or fingerprinting in call time. State scalar, ordered-history, edit/dependency, parent-effect and RNG verification separately.

| Pair | Reference distance | Rust distance | Verification / status | Ref. median s | Rust median s | Eligible speedup | Matched repeats |
|---|---|---|---|---|---|---|---|
| ResNet18 × Mixer d8 | — | — | NOT RUN | — | — | — | 0 |
| ResNet18 × Mixer d12 | — | — | NOT RUN | — | — | — | 0 |
| ResNet34 × Mixer d8 | — | — | NOT RUN | — | — | — | 0 |
| ResNet34 × Mixer d12 | — | — | NOT RUN | — | — | — | 0 |

| Pair | Reference baseline / peak MiB | Rust baseline / peak MiB | Reference peak delta MiB | Rust peak delta MiB | Retained RSS MiB (ref./Rust) | Status |
|---|---|---|---|---|---|---|
| ResNet18 × Mixer d8 | — | — | — | — | — | NOT RUN |
| ResNet18 × Mixer d12 | — | — | — | — | — | NOT RUN |
| ResNet34 × Mixer d8 | — | — | — | — | — | NOT RUN |
| ResNet34 × Mixer d12 | — | — | — | — | — | NOT RUN |

List raw repetitions, min/max and completion/exclusion counts separately. State RSS-reset status and any lifetime-only fallback labels. Do not compute a ratio from missing, mismatched, failed or censored values.

## 4. Off/on sensitivity within each backend

For every pair, report full-precision distance changes, selected-operation/history changes, status changes, and observed runtime/memory differences. Flag `0.25` is a fixed diagonal-offset threshold, not an asserted percentage reduction. Off/on equality is not a requirement.

## 5. Historical reproduction audit

Provide separate four-row tables for historically verified or candidate algorithm revisions, if run. Give unrounded distance, two-decimal rendering, formatting assumption and comparison with the printed value. Report the historical timing-boundary evidence. Do not treat selecting a matching revision or pruning mode as proof of historical provenance. Do not use historical screenshot time/new runtime as a controlled speedup.

## 6. Failures, limits and diagnostics

Include every construction error, grammar incompatibility, algorithm exception, semantic mismatch, timeout, RSS stop, worker error and budget-excluded configuration. Preserve progress stage, exit information and replay commands. A matching exception is not a successfully computed distance. Reverse/self pairs and model build/backward diagnostics remain separate from the primary four rows.

## 7. Conclusions — separate claims

Source definitions recovered: [evidence].

Runtime fixtures faithful: [evidence or remaining failures].

Python/Rust alignment equivalence: [scope/configurations].

Historical printed-distance/timing reproduction: [scope and unresolved evidence].

Controlled runtime/memory results: [only eligible measured comparisons].

## Reproduction artifacts

List exact rerun commands; fixture/runtime/source/binary/environment hashes; raw per-worker JSON/logs; relevant source changes/tests; campaign-budget ledger; reports and any figures. Do not overwrite historical targets with observations.
