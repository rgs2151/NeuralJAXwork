# Core concepts

## Modules describe computation

An `nn.Module` groups parameters, non-trainable buffers, child modules, and a
`forward` method. Calling `model(x)` runs that method. Composition can include
branches, residual additions, shared modules, and structured outputs. A model
has no built-in dataset or objective: those belong to the experiment.

Use `nn.Parameter` for differentiable weights and `nn.Buffer` for persistent
non-parameter state. Use ordinary attributes for immutable configuration. Module
paths give parameters and buffers stable names for inspection and checkpoints.

## Arrays and transformations

Inputs, outputs, and parameter values are JAX arrays. `nj.value_and_grad` returns
both a scalar objective and parameter gradients. `nj.jit` understands module
state and reference sharing, whereas raw `jax.jit` expects pure array functions.

A compiled step takes every stateful participant explicitly: model, optimizer,
and random streams. Keep dataset iteration, logging, file writes, and plotting
outside the step. Shapes and static configuration can trigger compilation; batch
padding, consistent shapes, and separate train/evaluation steps control that cost.

## Trainability and training mode are different

`model.train()` enables dropout and running-statistic updates; `model.eval()`
disables those training behaviors. Evaluation mode does not block gradients.
This matters when generator gradients pass through an evaluation-mode discriminator.
Optimizer ownership determines which parameters are updated.

## Randomness is state

Create parameter streams with `nj.Rngs(params=seed)` and runtime streams with
named keys such as `dropout` and `sample`. Drawing a key advances that stream.
Saving weights alone cannot reproduce the next random training step: save runtime
RNGs and optimizer state too. Evaluation figures should use a separate seed so
plotting does not change subsequent training.

## Data and objectives remain explicit

Data loaders produce batches; objective functions map predictions and targets to
a scalar. Cross-entropy expects raw logits and integer labels. Reconstruction and
forecasting losses compare matching-shaped arrays. Masked losses must normalize
by the number of valid items. Fit preprocessing on training data only and persist
feature order, vocabularies, split indices, and transformation statistics.

## What a checkpoint represents

A complete training checkpoint contains model variables, optimizer slots, RNG
state, progress, and a data-loader cursor. It restores a compatible, already
constructed Python architecture; it is not executable model source. A model's
`state_dict()` is a lighter weights-and-buffers snapshot.

The [API tutorials](api_tutorials.rst) demonstrate these concepts with small,
inspectable programs. The [model examples](tutorials.rst) apply them to datasets.
