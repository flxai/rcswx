# PyTorch guide

[Quickstart](../README.md#pytorch-example) · [Portable API](portable.md) ·
[Runnable examples](../examples/README.md)

Use `rcswx.torch` to build supported architecture trees into models, capture their
structure, and generate fresh offspring models. Install `rcswx[torch]`; the
[README example](../README.md#pytorch-example) is a complete starting point.

## Which models are supported?

There are two entry points:

- **RCSWX-built models:** `build` attaches data-only structural provenance, so
  `capture` can recover the architecture without guessing how it was constructed.
- **External models with an explicit converter:** register a converter for the
  supported module types before capture. The
  [converter example](../examples/torch_converter.py) demonstrates the deliberately
  narrow `nn.Sequential` Linear/ReLU converter.

This is not a generic importer for arbitrary `nn.Module`, FX, or ONNX graphs.
Unsupported structures are rejected, not approximated. TensorFlow/Keras
integration is not provided.

## Build a model from architecture data

`rcswx.torch.build(architecture, input_spec=..., build_options=...)` accepts an
`Architecture` or `CapturedArchitecture` and returns a fresh PyTorch module.
The installed builder supports the `einspace` and `hnasbench201` grammars at
version `"1"`. Operations and occurrence topology must be representable by the
grammar's callbacks; arbitrary node parameters are not a general layer
configuration interface.

Input assumptions describe what the grammar receives when reconstructing the
network. Supply them in the architecture's `input_spec`, or explicitly to
`build` if the architecture does not contain them. An explicit specification
must not conflict with stored assumptions. See the
[module example](../examples/torch_module_crossover.py) for an einspace input
specification with batch shape and feature mode.

Building initializes parameters and buffers afresh. It does not copy learned
weights, buffer values, gradients, or optimizer state, and it does not execute a
validation forward pass. Run the resulting model on representative inputs before
using it in an experiment.

## Cross two models

`rcswx.torch.crossover(first, second, seed=17)` captures the parents, performs
portable structural crossover, and builds a fresh child. Parents must have
matching grammar semantics and input assumptions. The parent modules' topology,
parameters, buffers, gradients, and training modes are left untouched.

Use `rcswx.torch.crossover_with_report` when you also need the child's portable
architecture and the crossover report. Its result exposes `child`,
`architecture`, `report`, and both captured parents. The
[report example](../examples/torch_module_crossover.py) prints the chosen
operations and runs the resulting model.

The module APIs do **not** transfer learned weights or buffer values. They do not
retry failed offspring or call the legacy `validated_crossover` API, and there is
no hidden validation forward pass. A valid structural selection does not imply
that every generated architecture has valid tensor shapes for your task.

### Device, dtype, and training mode

By default, module crossover uses compatible parent device/dtype settings and
parent one's **root** training mode. Ambiguous dtype/device layouts or
incompatible parent settings require explicit choices.

Pass `build_options=BuildOptions(device=..., dtype=..., training=...)`, or an
equivalent mapping. `device` chooses placement, `dtype` overrides floating/complex
tensor dtype, and `training` sets the training mode. These are execution choices,
not structural edit attributes.

### Structural seeds are not weight seeds

The native sampler uses RCSWX's own RNG. `seed=17` controls edit selection; it does
not seed PyTorch parameter initialization or consume NumPy's global RNG. Manage
PyTorch's RNG separately when you need repeatable initial weights. See
[sampling and replay](portable.md#sampling-and-replay) for streams, checkpoints,
and the optional reference numerical route.

## Capture and rebuild

`rcswx.torch.capture(model)` returns a `CapturedArchitecture`. Capture describes
architecture and execution settings, not optimizer state or an arbitrary
executable model. It never runs `forward`.

Capture rejects:

- Stale retained topology or bindings.
- Shared modules.
- Shared parameter or buffer objects.
- Distinct tensors that share backing storage.

Stale retained provenance is not silently reinterpreted through a converter.
External models need an explicitly registered converter; see
[`register_importer` and the converter example](../examples/torch_converter.py).

For retained RCSWX models, `build(capture(model))` preserves captured per-module
train/eval modes and per-tensor dtype, device, and gradient requirements, while
initializing fresh parameters and buffers. Explicit `BuildOptions` override
device, floating/complex dtype, or training mode. This architecture round trip is
not a replacement for saving trained model state.

## Save and load trained state

Use `rcswx.torch.save(model, path)` and `rcswx.torch.load(path)` for a managed
checkpoint containing a validated manifest and state dictionary. Unlike
`build(capture(model))`, loading restores the saved parameter and buffer values.
Optimizer state is not included.

Loading uses Torch's weights-only loader and rebuilds supported structures through
the installed grammar/converter contracts. It validates the manifest and state
dictionary rather than unpickling a whole executable model. A checkpoint is not
a way to import an otherwise unsupported architecture.

## Module APIs versus legacy tree APIs

`rcswx.torch.crossover` consumes modules and returns a fresh module. The root
`rcswx.crossover` API operates on architecture values or legacy callback-bearing
trees; tree crossover and model construction are separate steps.

For the legacy workflow, start with [parents.py](../examples/parents.py), then
[crossover.py](../examples/crossover.py). The
[validated crossover example](../examples/validated_crossover.py) adds
reconstruction and a validation forward pass. Legacy trees retain historical
mutation/ownership semantics; do not assume the module API's parent-isolation
contract applies to them. See [plan ownership](portable.md#plan-ownership-and-legacy-trees).

The implementation intentionally corrects some original operator behavior. See
[reference compatibility](compatibility.md) before comparing historical outputs.
