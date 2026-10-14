# Implementation roadmap and decision register

Historical implementation plan. Milestones record the original sequence of work;
[implementation status](IMPLEMENTATION.md) tracks delivered features and omissions.
Notebook execution is now local-only; CI renders saved outputs.

## Original architectural decisions

| Decision | Proposal | Reason / cost |
| --- | --- | --- |
| Model authoring | `nn.Module`, `__init__`, `forward`, registered children | Familiar composition; one base for all model families |
| Tensor semantics | Ordinary JAX arrays | Avoid building a second tensor/autograd engine |
| Gradient API | Explicit `value_and_grad` and `optimizer.step(grads)` | Fits JAX; differs from PyTorch backward syntax |
| Transformation | Graph-aware `nj.jit`; pure state boundary underneath | Must correctly preserve aliases, buffer updates, and RNGs |
| State backend | Prototype a Flax NNX adapter first | Reduces graph-engine work; introduces upstream dependency constraints |
| Optimizer backend | Evaluate Optax adapters behind stable contracts | Reuse update math; parameter identity mapping remains our responsibility |
| Initialization | Explicit dimensions and explicit RNG streams | Predictable construction and inspection; less lazy convenience |
| Image layout | NCHW public API | PyTorch familiarity; benchmark internal JAX layout costs |
| Sequence layout | Batch first, explicit recurrent carries | Consistent batching and streaming semantics |
| Orchestration | Optional task/trainer, steps own updates | One training contract, including GANs |
| Visualization | Versioned metadata and traces before renderers | Preserve inspectability without blocking core training |
| Compatibility | New API; no automatic legacy compatibility | Existing classes have different state and training semantics |

## Original design questions

1. **Backend commitment:** can a thin NNX adapter satisfy `forward`, explicit
   runtime RNGs, `ParameterSet` identity, joint graph transforms, and transactional
   updates? Prototype this before committing to the public names. If not, revise
   the public interface or implementation scope.
2. **Package identity:** use lowercase `neuraljaxwork` for new code? Decide whether
   the old uppercase package is archived, removed in a major version, or retained
   briefly with migration guidance. No migration is performed by these documents.
3. **Learning versus infrastructure:** how much layer math should be implemented
   directly with JAX primitives versus delegated to upstream layers? Recommended:
   use proven transformation machinery, implement and explain selected layers.
4. **First supported platforms:** choose a Python/JAX/Flax/Optax matrix after the
   backend prototype. Do not copy the legacy environment freeze or claim support
   for all GPUs. Pin a tested development environment and keep runtime dependencies
   minimal; documentation, plotting, and notebooks are extras.
5. **Trace fidelity:** define the first supported subset of JAX operations and
   control flow for computation schematics. Module summaries can ship before a
   complete operation-level trace.

The cookbook defines the proposed API; these questions record the implementation
choices considered before version 0.1.0.

## Milestone 0: prove the state model

Prototype outside the legacy API before a broad rewrite. Demonstrate a two-layer
module, shared parameter references, explicit RNGs, a mutable buffer, and an
optimizer referencing the same parameters through a compiled step. Verify:

- Eager and compiled outputs, gradients, and updates agree within dtype tolerance.
- Buffer/RNG updates occur exactly once under differentiation and compilation.
- Tied weights remain tied and receive the summed gradient once.
- A train/eval switch cannot reuse stale mode behavior.
- Invalid transformed calls do not partially commit state.
- A save/restore roundtrip resumes the next update identically on the same backend.

If this fails, revise the core design before adding architecture families.

## Milestone 1: trustworthy dense training

Implement `Module`, parameters/buffers, containers, RNGs, Linear, activations,
losses, SGD/Adam, data batches, manual loops, and a minimal state checkpoint.
Deliver deterministic XOR and a small dense MNIST run.

Acceptance: loss/gradient checks against independent calculations, finite-difference
checks at differentiable points, correct loss reductions, parameter registration
and alias tests, deterministic initialization, and actual learning assertions.
An example exiting successfully is insufficient. Establish numerical tolerances
and learning thresholds from measured baselines.

## Milestone 2: CNNs and inspection foundations

Add convolutions, pooling, normalization, dropout, shape diagnostics, module
summaries, ModelSpec, structured run logs, and full training checkpoints. Deliver
MNIST CNN and Titanic examples before a longer cats/dogs showcase.

Acceptance: forward and gradient comparisons for layout/padding/stride/groups;
BatchNorm train/eval checks; dropout RNG reproducibility; strict checkpoint mismatch
errors; short CPU examples in CI; single-accelerator checks when hardware exists.
Benchmark only after correctness, separating compilation time from steady-state
time and synchronizing device work before reporting timings.

## Milestone 3: recurrent models

Implement RNNCell/RNN and LSTMCell/LSTM using scan-compatible state. Add padded
sequence collation and length masks. Deliver a controlled RNN/LSTM comparison.

Acceptance: scan agrees with a simple reference loop; states and sequence shapes
are correct; padding cannot alter final valid states; zero-length policy works;
chunked inference matches unchunked inference; truncated-gradient boundaries are
tested. Bidirectionality and recurrent dropout can follow basic correctness.

## Milestone 4: transformers

Add embeddings, LayerNorm, attention, mask helpers, pre-norm transformer examples,
and a small next-token task. Later add cross-attention and cached generation.

Acceptance: no future-token leakage under causal masks; masked keys cannot affect
valid outputs; fully masked rows remain finite and zero; head dimensions validate;
loss uses shifted targets and valid counts. When cache support arrives, cached and
uncached generation logits must agree for the same prefix.

## Milestone 5: generative examples and orchestration

Dense autoencoders and GANs are already expressible after milestone 1; they may
be prototyped earlier. At this milestone, formalize generative recipes, transpose
convolutions, multiple-optimizer checkpoints, and the optional task/trainer layer.

Acceptance: reconstructions match target shape; generator gradients flow through
the discriminator; discriminator steps cannot change generator parameters and
vice versa; inactive buffers follow the documented mode policy; resume restores
both optimizers; trainer and manual loops produce matching updates. Check sample
quality empirically without promising GAN convergence.

## Milestone 6: notebook gallery and visual tooling

Build the notebook sequence in 04-inspection.md, architecture renderers, bounded
intermediate capture, and artifact-based plots. Deliver a consistent set of real,
reproducible demonstrations.

Acceptance: notebooks execute locally in a supported environment; CI validates
saved outputs without execution; full runs record hardware, duration, data versions, and checkpoints;
plots regenerate from artifacts; shared modules and residual edges render
accurately; tracing/capture cannot perturb subsequent training.

## Proposed eventual layout

```text
design/                         # these documents; decisions and API sketches
src/neuraljaxwork/
  core/                         # graph/state adapter, variables, RNGs, transforms
  nn/
    module.py
    containers.py
    linear.py
    convolution.py
    normalization.py
    recurrent.py
    attention.py
    functional/                 # stateless operations and losses
  optim/                        # optimizers, parameter groups, clipping, schedules
  data/                         # datasets, collation, sampling, loading
  metrics/                      # reducers and task metrics
  checkpoint/                   # schemas, manifests, atomic save/restore
  inspection/                   # summaries, ModelSpec, traces, capture
  experiments/                  # local run records and artifact metadata
  training/                     # optional Task, Trainer, callbacks
  visualization/                # optional renderers, no core dependency
examples/                       # reusable dataset/training/evaluation programs
notebooks/                      # explanatory frontends over examples
tests/                          # mathematical, state, integration, and smoke tests
docs/                           # future user guide and generated API reference
pyproject.toml                  # tested runtime dependencies and optional extras
```

This layout is a proposal; the implemented package uses a more compact structure.
Avoid one framework class per architecture name: MLPs, autoencoders, GANs, and
transformer applications primarily belong in examples or a later models library.
Reusable layers and state/control contracts belong in the core package.

## Reference points

Official documentation consulted on 2026-10-14. These are inspirations, not claims
that the proposed facade already exists or is source-compatible with them.

- [PyTorch Module](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html)
  and [module notes](https://docs.pytorch.org/docs/stable/notes/modules.html):
  module composition, registered parameters/buffers, mode switching, and state
  dictionaries inform the authoring experience.
- [Flax NNX basics](https://flax.readthedocs.io/en/stable/nnx_basics.html):
  object-oriented modules, explicit state/randomness, reference sharing, and
  graph-aware transformations inform the JAX execution strategy. Flax evolves;
  the prototype must target a tested version rather than assume internals are stable.
- [JAX sharp bits](https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html):
  pure-function expectations, random-key handling, and tracing constraints motivate
  explicit state boundaries and restrictions on Python side effects.
- [Lightning optimization](https://api.lightning.ai/docs/pytorch/stable/common/optimization)
  and [basic model training](https://api.lightning.ai/docs/pytorch/stable/reference/model/train_model_basic):
  task-level training logic and manual optimization inspire optional orchestration.
  Our proposed trainer always delegates updates to the task step.

Release-specific decisions and deviations are recorded in
[implementation status](IMPLEMENTATION.md).
