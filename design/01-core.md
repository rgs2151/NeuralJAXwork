# Core module and state contract

Historical proposal; see [implementation status](IMPLEMENTATION.md) for supported
interfaces and differences in version 0.1.0.

## Modules and registration

Every trainable architecture inherits from `nn.Module`. Built-in layers and custom
composites use the same base. `forward(*args, **kwargs)` is user-defined;
`__call__` dispatches it and supplies framework bookkeeping. Users call `model(x)`,
not `model.forward(x)`, so inspection can observe module boundaries.

```python
class ResidualBlock(nn.Module):
    def __init__(self, width, *, rngs):
        super().__init__()
        self.fc1 = nn.Linear(width, width, rngs=rngs)
        self.fc2 = nn.Linear(width, width, rngs=rngs)

    def forward(self, x):
        return x + self.fc2(F.relu(self.fc1(x)))
```

Assignment registers child modules, `Parameter`, and `Buffer` objects. A parameter
exposes its array as `.value`; a raw array attribute is rejected with guidance to
wrap it as a parameter or buffer. Python scalars, tuples, and immutable configuration
objects are static metadata. Use `ModuleList`, `ModuleDict`, `ParameterList`, and
`ParameterDict` for registered collections; putting modules in ordinary lists or
dictionaries must raise a clear error rather than silently omit parameters.

`nn.Sequential` chains single-input/single-output modules. It is a convenience,
not a separate model type. Branches and multiple outputs belong in `forward`.
`ModuleList` iterates modules but does not define a forward pass.

| API | Proposed contract |
| --- | --- |
| `model.parameters()` | Live `ParameterSet` of trainable parameter references, deduplicated by identity |
| `model.named_parameters()` | Stable canonical path and parameter pairs |
| `model.buffers()` / `named_buffers()` | Non-parameter state |
| `model.children()` / `named_modules()` | Immediate children / recursive module traversal |
| `model.train(mode=True)` / `eval()` | Recursively set training mode; return `self` |
| `model.state_dict()` | Detached snapshot of parameters and persistent buffers, with schema and alias metadata |
| `model.load_state_dict(state, strict=True)` | Validate and copy state into an existing architecture |
| `model.to(device=..., dtype=...)` | Move/cast arrays outside compiled steps; default before optimizer creation |

Shared references are legal: assigning the same module to two fields ties weights.
Updates, parameter counts, checkpoints, and gradients must deduplicate aliases.
Canonical paths are deterministic, with an alias table for other paths. Replacing
or adding parameters invalidates existing optimizers and compiled graph signatures;
rebuild them explicitly. Changing values without changing structure does not.

## Initialization and randomness

Dimensions are explicit at construction. No lazy initialization in the first
version: `nn.Linear(in_features, out_features, rngs=rngs)` creates parameters
immediately. Default weights use a documented fan-in initializer; biases default
to zero. Initializer choices and logical weight shapes appear in module config.

```python
rngs = nj.Rngs(params=42)
a = nn.Linear(32, 32, rngs=rngs)
b = nn.Linear(32, 32, rngs=rngs)  # different draws from the same stream
runtime_rngs = nj.Rngs(dropout=43, sample=44)
```

Each call `rngs.next("dropout")` returns a new key and advances that stream's
counter. No global seed or implicit default stream. Missing streams are errors.
Initialization streams and runtime streams are separate; modules do not retain
constructor RNG references. A stochastic forward accepts `rngs` explicitly and
passes it to stochastic children. Dropout in evaluation returns its input without
consuming a key. The same seed, construction order, and execution order reproduce
the same sequence on a supported backend; bitwise equality across devices is not
promised. Distributed stream folding is a later, separately specified feature.

## Shapes, outputs, and layers

Use batch-first conventions throughout:

| Operation | Input / output convention |
| --- | --- |
| `Linear(I, O)` | `[..., I] -> [..., O]`; stored weight `[O, I]` |
| `Conv2d(I, O, ...)` | `[B, C, H, W]`; logical weight `[O, I/groups, KH, KW]` |
| `Embedding(V, D)` | Integer indices `[...] -> [..., D]` |
| `RNN` / `LSTM` | `[B, T, D]`; hidden state `[layers * directions, B, H]` |
| Attention | `[B, T, D]`; heads are internal |
| Classification | Raw logits `[..., classes]`; integer targets `[...]` |

Convolutions present channels-first tensors for familiarity. Any internal layout
conversion is explicit in the implementation and benchmarked; performance is not
assumed. Loss class-axis convention is **last axis**, including sequence logits.
Segmentation callers must move the channel axis or specify `class_axis=1`.

Outputs can be arrays, tuples, dictionaries, or registered pytree dataclasses.
Prefer named dictionaries for auxiliary information; their keys and structure stay
fixed within a compiled signature. All layers define shape validation, dtype rules,
initialization, mutable buffers, RNG streams, and serialization config.

Initial catalog: `Linear`, `Embedding`, `ReLU`, `GELU`, `Tanh`, `Sigmoid`, `Dropout`,
`Flatten`, `Sequential`, registered containers, `Conv1d/2d`, `ConvTranspose2d`,
`MaxPool2d`, `AvgPool2d`, `AdaptiveAvgPool2d`, `BatchNorm1d/2d`, `LayerNorm`,
`RNNCell`, `RNN`, `LSTMCell`, `LSTM`, `MultiheadAttention`, and transformer blocks.
This is a roadmap, not the minimum first release.

## Autodiff and optimizer ownership

`nj.value_and_grad(fn, argnums=0, has_aux=False)` differentiates only trainable
`Parameter` leaves of the selected module argument. With `has_aux=True`, `fn`
returns `(scalar_loss, auxiliary_pytree)` and the transform returns
`((loss, auxiliary_pytree), Gradients)`. Gradient structure records parameter paths
and identity relationships and must match an optimizer's `ParameterSet`.

There is no persistent `.grad`, `zero_grad()`, or `loss.backward()`. JAX arrays
remain ordinary JAX arrays. Explicit gradient values make accumulation, clipping,
multiple objectives, and separate optimizers easier to reason about.

`optim.Adam(model.parameters(), lr=...)` retains references to that parameter set
and owns moments and step counters. `optimizer.step(grads)` updates those parameters
once. Shared references between optimizer and model must survive transforms.
An unrelated gradient tree, duplicate optimizer ownership within one step, or
stale structure must raise a useful error. Two GAN optimizers own disjoint sets.
Advanced overlapping optimizers require a future explicit contract.

Frozen parameters are excluded from differentiation and optimizer ownership using
`parameter.requires_grad_(False)` before creating the optimizer. Freezing a module
does not imply `eval()`: trainability and running-statistics behavior are separate.
Input gradients remain possible through frozen weights.

## Mutation and compilation: the central engineering requirement

`nj.jit` is graph-aware, not an alias for `jax.jit`. It converts module, optimizer,
and RNG state into explicit arrays for the compiled function, then commits returned
state to the original objects. A joint graph preserves parameter sharing between
model and optimizer. Supported state includes parameters, buffers, optimizer slots,
and RNG counters. Arbitrary list appends, filesystem writes, printing, and Python
side effects are not compiled state updates.

`value_and_grad` commits non-parameter updates from the primal forward exactly
once; autodiff must not double-update BatchNorm statistics or RNG counters. It
does not change parameter values. `optimizer.step` performs that change. Repeated
forwards intentionally produce repeated state updates unless using evaluation mode.
An exception during a transformed call must leave caller state uncommitted.

All mutable participants must be explicit transformed arguments. A local objective
may close over other arguments of the enclosing transformed step, as in GAN
examples, but not untracked mutable globals. Gradients select the candidate module;
other arguments are constants for differentiation while supported state updates
remain tracked by the outer step.

Mode switches happen outside a compiled step and participate in its static
signature. Different shapes, modes, and static configurations may recompile.
Use separate train/eval steps. Ordinary Python loops over a static `ModuleList`
are supported; data-dependent flow uses JAX control-flow primitives through
graph-aware wrappers. Recurrent sequence loops lower to scan rather than unrolling
Python per timestep. Variable-length batches use padding, masks, or bucketing.

For raw JAX interoperability, expose a pure boundary:

```python
graph, state = nj.split(model)       # state: params, buffers, mode metadata
output, next_state = nj.apply(graph, state, x)
nj.update(model, next_state)         # explicit host-side commit
```

`graph` is static structure, including aliases; array state is a pytree. Stochastic
pure application additionally takes `rng_state` and returns `next_rng_state`.
Joint `split((model, optimizer, rngs))` preserves cross-object aliases and is the
basis of graph-aware transforms. The exact lowering interface needs a prototype
before this convenience API is frozen. Raw `jax.jit(model)` is not supported.

Recommended implementation direction: evaluate Flax NNX as the module/state and
transform backend, and Optax for optimizer mathematics. Keep NeuralJAXwork's
public contracts and adapters explicit. A custom graph engine is a much larger
project and is not a prerequisite for learning to implement layers and models.

## Checkpoints and architecture reconstruction

Distinguish three artifacts:

1. `state_dict`: weights and persistent buffers for an existing model.
2. `ModelSpec`: versioned JSON-compatible architecture config and alias graph.
3. Training checkpoint: model state, optimizer state, RNG state, progress, data
   sampler state, and experiment metadata.

Built-ins implement `get_config()` and a registered reconstruction factory.
Custom models can register `type_id`, schema version, `get_config()`, and
`from_config(config, rngs=...)`. State is loaded after constructing the graph.
Arbitrary Python `forward` source cannot be safely reconstructed from weights or
a diagram: custom code must remain importable at a recorded revision. Loading a
checkpoint does not execute embedded Python. Unknown type IDs require explicit
registration. Strict loads validate paths, shapes, dtypes, aliases, and versions
before mutating any state.
