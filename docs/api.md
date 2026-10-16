# API and semantics

## Modules

Subclass `nn.Module`, call `super().__init__()`, define children in the constructor,
and implement `forward`. Call modules as `model(x)` so composition stays consistent.
A child can return arrays, tuples, or dictionaries of arrays.

`nn.Parameter(array)` registers a differentiable variable; `nn.Buffer(array)`
registers a non-parameter state variable. Access arrays through `.value`. Assigning
the same module to two attributes shares weights. Parameter traversal deduplicates
by identity. `ModuleList` and `ModuleDict` express named collections; the underlying
NNX backend also discovers ordinary lists and dictionaries.

`named_parameters()`, `named_buffers()`, and `named_modules()` produce canonical
paths. `parameters()` returns a live selection for an optimizer, not copies.
`state_dict()` returns independent NumPy snapshots. A strict `load_state_dict`
checks paths, shape, and dtype before copying values. Structural edits require a
new optimizer and compiled step. Do not move/cast parameters after creating an
optimizer; optimizer moments would otherwise retain the old device/dtype.

`train()`/`eval()` recursively toggle stochastic/normalization behavior and return
self. They do not turn autodiff on/off. Switch mode outside a compiled step.

## Layers and shapes

| Layer | Signature and convention |
| --- | --- |
| `Linear(I, O, bias=True, rngs=...)` | `[..., I] -> [..., O]`, weight `[O,I]` |
| `Conv1d(I, O, K, stride=1, padding=0, groups=1, rngs=...)` | `[B,C,L]`; integer padding |
| `Conv2d(I, O, K, stride=1, padding=0, groups=1, rngs=...)` | NCHW; integer/pair or SAME/VALID padding |
| `ConvTranspose2d(I, O, K, stride=1, padding=0, output_padding=0, rngs=...)` | NCHW; grouping and dilation unsupported; weights `[O,I,KH,KW]` |
| `MaxPool2d(K, stride=None)`, `AvgPool2d(...)` | NCHW, valid windows |
| `AdaptiveAvgPool2d((1,1))` | Global spatial mean; other sizes rejected |
| `BatchNorm1d(C)`, `BatchNorm2d(C)` | NC/NCL or NCHW; affine parameters and running buffers |
| `LayerNorm(shape)` | Normalize trailing dimensions |
| `Dropout(p)` | Explicit `rngs` in training; identity in evaluation |
| `Embedding(V,D,rngs=...)` | Integer `[...] -> [...,D]` |
| `RNN(I,H,num_layers=1, rngs=...)` | `[B,T,I] -> (sequence, h)` |
| `LSTM(I,H,num_layers=1, rngs=...)` | `[B,T,I] -> (sequence, (h,c))` |
| `MultiheadAttention(D, heads, rngs=...)` | Query `[B,Tq,D]`, key/value `[B,Tk,D]` |

ReLU, GELU, Tanh, Sigmoid, Flatten, and Sequential provide simple composition.
`Sequential` forwards one value without RNG keywords; write a custom `forward`
for stochastic layers or multiple inputs. Linear/conv use fan-in-scaled uniform
weights and zero biases. LSTM forget bias is one. BatchNorm uses population batch
variance for both normalization and running-statistic updates, not PyTorch's
unbiased running-variance update. Results can therefore differ from PyTorch.

Transposed convolution output length on each spatial axis is
`(input - 1) * stride - 2 * padding + kernel_size + output_padding`.

## Recurrent state

Hidden states use `[layers,B,H]`. `state=None` initializes zeros. LSTM state is a
pair `(h,c)`. `lengths[B]` freezes the carry after valid timesteps and zeros padded
outputs. Zero-length examples keep their initial carry. Lengths must be integers
between zero and T; validate data before entering compiled code. Bidirectionality
is not implemented. For streaming, pass the returned carry to the next chunk;
for truncated backpropagation, call `js.stop_gradient(carry)` between chunks.

## Attention

Masks are boolean and broadcast to `[B,heads,Tq,Tk]`; **True means allowed**.
`nn.attention.causal_mask(T)` has shape `[1,1,T,T]` and
`padding_mask(valid[B,T])` has shape `[B,1,1,T]`. Padded query positions still need
a loss mask. Fully masked rows return zero weights and output, including projection
bias, with finite gradients. `need_weights=True` returns `(output, weights)` and
must be static under compilation. Width must be divisible by heads.
Embedding indices and class targets must be in range; these are input-data
preconditions, not runtime checks inserted into compiled kernels.

## Randomness and compilation

Constructors consume `js.Rngs(params=seed)`. Runtime dropout consumes a separate
`js.Rngs(dropout=seed)`. `rngs.next(name)` advances a named checkpointable stream;
missing names are errors. Sampling can use `js.random` (JAX's random namespace).

`js.jit`, `js.value_and_grad`, and `js.grad` are graph-aware NNX transforms.
Pass stateful objects explicitly. Use ordinary JAX transforms only on pure array
functions or the `split/merge/apply/update` boundary. Static shapes/modes can
recompile. Avoid arbitrary Python side effects inside a transformed function.
`value_and_grad` differentiates parameter leaves of the selected module and
returns an NNX gradient state; `has_aux=True` returns `((loss, aux), grads)`.

NNX semantics govern exceptions and advanced transform nesting. Eager mutations are not automatically rolled back after an exception. Keep shapes/configuration valid before calling transformed code.

## Losses and optimization

`nn.functional` exposes `cross_entropy`, `binary_cross_entropy_with_logits`,
`mse_loss`, `l1_loss`, and `smooth_l1_loss`. Prediction/target shapes must match
except cross-entropy's class axis. Use integer labels with raw class logits;
class axis defaults to last. Reductions are `none`, `sum`, `mean`.
Masked cross-entropy divides by valid count, with zero loss/gradient for an
all-masked batch. BCE uses logits for numerical stability.

`optim.SGD`, `Adam`, and `AdamW` accept `model.parameters()` and `lr`.
`optimizer.step(grads)` updates once; `optimizer.step_count` reports the counter.
`clip_grad_norm(grads, max_norm)` returns `(clipped_grads, original_norm)`.
There are no `.grad` buffers, implicit zeroing, parameter groups, or freezing API
in this release. Use disjoint modules/optimizers for GANs and differentiate only
the intended module. Gradients must originate from the same model structure;
matching-shaped trees from an unrelated model are not identity-validated.

## Training utilities

`training` includes compiled steps for classification, CNN dropout, forecasting,
autoencoding, and the two GAN phases. They are readable reference implementations,
not a mandatory trainer abstraction. Use your own step for custom objectives.

`Task(model, optimizer, rngs=None)` may override `training_step`,
`validation_step`, `set_phase`, and `checkpoint_objects`. Steps own updates;
Trainer never differentiates again. `Trainer(max_epochs, callbacks=(), run=None)`
aggregates `loss`, `count`, and optional `correct`; custom metrics/GAN orchestration
should use the explicit loop. `ModelCheckpoint(path, monitor='val/loss', mode='min')`
saves the selected checkpoint. Automatic trainer resume and callback-state resume
are not implemented; use the explicit checkpoint/loader APIs.
