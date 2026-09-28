# PyTorch guide

[Quickstart](../README.md#pytorch-example) · [Portable API](portable.md) ·
[Runnable examples](../examples/README.md)

Install `rcswx[torch]`. These recipes build and cross models, run a training step,
and save the resulting state. The snippets through the checkpoint recipe run in
order; the external-model recipe uses the same `input_spec`.

## Which models are supported?

- **Ordinary PyTorch models in the supported grammar:** `import_model` captures
  their computation, lowers it into the existing grammar, and verifies a
  reconstruction against the original.
- **RCSWX-built models:** `build` retains structural provenance, so `capture`
  can recover the architecture without guessing how it was constructed.
- **External models with an explicit converter:** register a converter for the
  supported types. The [Linear/ReLU recipe](#import-a-supported-nnsequential-model)
  below is the legacy, no-forward path for a narrow `nn.Sequential` subset.

Unsupported structures are rejected, not approximated. This is not arbitrary
FX/ONNX graph execution, and TensorFlow/Keras integration is not provided.

## Import an ordinary PyTorch model

```python
import torch
from torch import nn
from rcswx.torch import build, import_model

source = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 32))
examples = (torch.randn(4, 8),)
imported = import_model(
    source, example_inputs=examples, grammar="einspace", grammar_version="1"
)
fresh_model = build(imported)
```

`import_model` traces the computation with FX on an isolated, meta-device copy;
it does not infer execution order from the module hierarchy. It returns a
data-only `CapturedArchitecture`, not the source model or its learned tensors.
`build(imported)` initializes fresh state: **verification is not inheritance**.

Grammar selection is explicit and defaults to the existing `einspace` version
`"1"` (the grammar originating in einsearch). Other names or versions are
rejected. Neither the grammar nor native RCSWX alignment, edit costs, or sampling
is changed by importing a model.

The supported language is deliberately narrower than PyTorch:

- Biased `Linear` with output widths 16, 32, 64, 128, 256, 512, 1024, or 2048.
  `F.linear` is accepted only with registered weight and bias parameters.
- Non-inplace ReLU, identity, and softmax on the final dimension.
- Rank-two `LayerNorm` over the final dimension, rank-three `BatchNorm1d`, and
  rank-four `BatchNorm2d`, with the grammar's default epsilon, affine state,
  momentum, and running-statistics behavior.
- Biased, ungrouped, zero-padded `Conv2d` with dilation one, supported output
  widths, and square `(kernel, stride, padding)` tuples `(1,1,0)`, `(1,2,0)`,
  `(3,1,1)`, `(3,2,1)`, `(4,4,0)`, `(8,8,0)`, or `(16,16,0)`. This lowers to
  the existing image-to-column, linear, and column-to-image routing operations.
- Sequential composition and nested, structured two-branch forks with equal-shape
  addition or concatenation on a grammar-supported non-batch dimension. Branch
  order is preserved. Crossing dependencies, repeated stateful invocations, and
  shared modules/storage are rejected rather than duplicated.

Module, functional, and method spellings are normalized only where their
semantics match these operations. Singleton computations receive the grammar's
required sequential/identity root embedding. Every resulting derivation must
also satisfy the existing grammar's layout and shape rules; this is not a promise
that arbitrary combinations of individually supported layers are representable.

Supply exactly one finite, floating-point, rank-two to rank-four CPU/CUDA tensor.
The capture records its fixed shape, dtype, and device. Mixed module modes,
hooks, unsupported options, data-dependent control flow, unrepresented work,
and detected forward-time configuration mutation are errors.

Import always validates grammar/shape constraints and a one-to-one state
correspondence, then compares original and reconstructed outputs and buffer
updates on disposable real-tensor copies in both training and evaluation modes.
Corresponding state is copied **only into the verifier**; convolution weights are
reshaped into the grammar's linear representation there. Tolerances are
dtype-aware and recorded in `metadata["validation"]`; nonfinite outputs fail.
The caller's tensors, parameters, gradients, buffers, modes, and PyTorch RNG
state are preserved on success and failure.

These checks establish faithfulness for the supported lowering and supplied
example, not equivalence of arbitrary Python programs or unseen input signatures.
Custom Python code executes during tracing/copying: it must not perform external
side effects, and this API is not an untrusted-code sandbox. Opaque/custom-block
exchange requires a future, more expressive grammar; there is no opaque fallback
in `einspace` version `"1"`.

Existing `capture`, `build`, `crossover`, and `crossover_with_report` remain
no-forward operations. Only the explicit import/validation workflow executes
isolated verification forwards.

## Build a model from architecture data

This pair differs in both output width and activation. Define the architectures,
then build normal PyTorch modules:

```python
import torch
from rcswx import Architecture
from rcswx.torch import build


def architecture(width, activation):
    return Architecture.from_tree(
        (
            "sequential",
            ("computation", (f"linear({width})",)),
            ("computation", (activation,)),
        )
    )


input_spec = {
    "shape": [4, 8],
    "mode": "col",
    "other_shape": None,
    "other_mode": None,
    "branching_factor": 1,
    "last_im_shape": None,
}

torch.manual_seed(0)
first_model = build(architecture(16, "relu"), input_spec=input_spec)
second_model = build(architecture(32, "softmax"), input_spec=input_spec)
```

Each tuple is `(operation_name, *children)`. In the default `einspace` version
`"1"` grammar, `sequential` composes two subtrees; `computation` wraps a layer;
`linear(16)` specifies 16 **output** features. Input width comes from
`input_spec`, not from the operation name. These parents behave as
`Linear(8, 16) → ReLU` and `Linear(8, 32) → Softmax`.

To change a parent's width or activation, change the arguments to `architecture`.
The installed builder supports the `einspace` and `hnasbench201` grammars at
version `"1"`, not arbitrary operation names. Operations and occurrence topology
must be representable by the grammar's callbacks; node parameters are not a
free-form `nn.Module` constructor interface. See the
[portable representation recipe](portable.md#represent-an-architecture) for the
tree format and the [einspace grammar](../python/rcswx/grammars/einspace.py) for
available operations.

### Adapt the input specification

For a dense network, set `shape` to `[batch_size, input_features]` and keep the
other values shown above. The six fields are grammar input assumptions, not
PyTorch constructor arguments:

| Field | Meaning | Value for this dense example |
| --- | --- | --- |
| `shape` | Input tensor dimensions, including the batch dimension. | `[4, 8]`: four examples, eight features each. |
| `mode` | Layout: `"col"` uses the last dimension for features; `"im"` describes image layout. | `"col"`. |
| `other_shape` | Shape of the other branch when branches are combined. | `None`: no second input branch. |
| `other_mode` | Layout of that other branch. | `None`. |
| `branching_factor` | Branch multiplicity used by grammar shape inference. | `1`: unbranched input. |
| `last_im_shape` | Spatial dimensions retained by image-to-column routing for conversion back to image layout. | `None`: no preceding image-to-column conversion. |

For channel-first images, `shape` is `[batch, channels, height, width]` and the
input mode is `"im"`; the tree must also use operations appropriate to that layout.
Changing only the shape does not turn this dense example into a CNN. The grammar
propagates branch and layout state through intermediate operations. Supply all
six input fields, even when their initial values are `None`.

Input assumptions can instead be stored in `Architecture.from_tree(...,
input_spec=input_spec)`. An explicit specification passed to `build` must not
conflict with stored assumptions. `build` accepts either an `Architecture` or a
`CapturedArchitecture`.

Building initializes parameters and buffers afresh. It does not copy learned
state or run a validation forward pass. Always exercise the model on
representative input before using it in an experiment.

## Cross two models

Continue with the two modules above. Use the report to inspect what actually
changed, rather than assuming every sampled child mixes the parents:

```python
from rcswx.torch import crossover_with_report

result = crossover_with_report(first_model, second_model, seed=0)
child_model = result.child
print("Selected edits:", result.report["crossover_operations"])
print(child_model)
child_model.eval()
with torch.no_grad():
    print("Output shape:", tuple(child_model(torch.randn(4, 8)).shape))
```

For this pair, the selected activation edit costs `0.5`. The child combines
parent two's `Linear(8, 32)` with parent one's `ReLU`, and produces shape `(4, 32)`.
Other selections can reproduce either parent's structure; all module offspring
still have fresh parameters and buffers. If you only need the module, use
`rcswx.torch.crossover(first_model, second_model, seed=0)`.

Parents must have matching grammar semantics and input assumptions. Their
topology, parameters, buffers, gradients, and training modes remain untouched.
The result exposes `child`, its portable `architecture`, the `report`, and both
captured parents as `parent1` and `parent2`.

There is no hidden validation forward pass, weight inheritance, or retry loop.
The module API does not call legacy `validated_crossover`. A valid structural
selection does not guarantee valid tensor shapes for every task.

### Train the child with PyTorch

The child supports the usual module and optimizer APIs. Here is one step on a
synthetic regression batch; replace the data and loss with your own task:

```python
child_model.train()
optimizer = torch.optim.AdamW(child_model.parameters(), lr=1e-3)
inputs = torch.randn(4, 8)
targets = torch.rand(4, 32)
optimizer.zero_grad()
loss = torch.nn.functional.mse_loss(child_model(inputs), targets)
loss.backward()
optimizer.step()
```

Training, evaluation, population management, and deciding whether an offspring
is useful remain caller-owned.

### Device, dtype, and training mode

By default, module crossover uses compatible parent device/dtype settings and
parent one's **root** training mode. Ambiguous dtype/device layouts or
incompatible parents require explicit choices, for example
`build_options={"device": "cpu", "dtype": torch.float32, "training": False}`.
`BuildOptions` accepts the same fields. `dtype` overrides floating/complex tensor
dtype. These are execution choices, not structural edit attributes.

### Structural seeds are not weight seeds

`seed=0` controls edit selection through RCSWX's native RNG; `torch.manual_seed(0)`
controls PyTorch initialization and the example's random tensors. Neither seeds
the other. Native sampling does not consume NumPy's global RNG. See
[sampling and replay](portable.md#sampling-and-replay) for streams, checkpoints,
and the optional reference numerical route.

## Capture and rebuild

To recover the structure of a supported model and create a fresh initialization:

```python
from rcswx.torch import capture

captured = capture(child_model)
rebuilt = build(captured)
```

Capture returns a `CapturedArchitecture` without running `forward`. For retained
RCSWX models, rebuilding preserves captured per-module train/eval modes and
per-tensor dtype, device, and gradient requirements, **not tensor values**.
Explicit build options override device, floating/complex dtype, or training mode.

Capture rejects stale retained topology or bindings, shared modules, shared
parameter/buffer objects, and distinct tensors that share backing storage.
Stale provenance is not silently reinterpreted through a converter.

## Save and load trained state

Use a managed checkpoint when you want to keep the trained values, rather than
just reconstruct the architecture:

```python
from rcswx.torch import load, save

child_model.eval()
save(child_model, "child.pt")
restored = load("child.pt")
inputs = torch.randn(4, 8)
with torch.no_grad():
    torch.testing.assert_close(restored(inputs), child_model(inputs))
```

The checkpoint contains a validated manifest and state dictionary. Unlike
`build(capture(model))`, loading restores parameter and buffer values. Optimizer
state is not included; save it separately if you need to resume training.

Loading uses Torch's weights-only loader and rebuilds supported structures through
the installed grammar/converter contracts. It validates the manifest and state
dictionary rather than unpickling a whole executable model. A checkpoint cannot
make an unsupported architecture importable.

## Import a supported nn.Sequential model

Using the `input_spec` from above, this recipe starts from familiar PyTorch code:

```python
from torch import nn
from rcswx.torch import register_importer, unregister_importer
from rcswx.torch.converters import linear_relu_sequential_converter

source = nn.Sequential(nn.Linear(8, 16), nn.ReLU())
register_importer(
    "linear-relu-sequential", "1", [nn.Sequential], linear_relu_sequential_converter
)
try:
    captured_source = capture(source, input_spec=input_spec)
    fresh_model = build(captured_source)
    with torch.no_grad():
        print("Rebuilt output shape:", tuple(fresh_model(torch.randn(4, 8)).shape))
finally:
    unregister_importer("linear-relu-sequential", "1")
```

The bundled converter accepts **exactly** `nn.Sequential(nn.Linear(...),
nn.ReLU())`: two modules, a biased linear layer, a non-inplace ReLU, and no
subclasses. The input feature width must match `input_spec`; the output width
must be supported by the grammar to rebuild it. It does not import deeper
sequences, arbitrary activation functions, or whole trained networks. Imported
architecture capture does not make `build` copy source weights.

Register converters explicitly before capture or module crossover, and keep
registration active while importing the models. See the
[standalone converter script](../examples/torch_converter.py) for the complete
setup. Other model families require a converter that describes their structure
and bindings, not just registering them with this narrow converter.

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
