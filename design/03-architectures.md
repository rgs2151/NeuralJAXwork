# Architecture cookbook

Status: proposed API examples, not executable against the current repository.
All examples share these imports and the core contracts; models do not acquire
special training methods based on their architecture family.

```python
import jax.numpy as jnp
import neuraljaxwork as nj
from neuraljaxwork import nn, optim
from neuraljaxwork.nn import functional as F
```

## Dense networks: tabular classification and MNIST

```python
class MLP(nn.Module):
    def __init__(self, in_features, hidden_sizes, out_features, *, rngs):
        super().__init__()
        widths = [in_features, *hidden_sizes]
        self.hidden = nn.ModuleList([
            nn.Linear(a, b, rngs=rngs)
            for a, b in zip(widths[:-1], widths[1:])
        ])
        self.head = nn.Linear(widths[-1], out_features, rngs=rngs)

    def forward(self, x):
        for layer in self.hidden:
            x = F.relu(layer(x))
        return self.head(x)

titanic = MLP(12, [64, 32], 1, rngs=nj.Rngs(params=1))
# 12 is illustrative: derive the actual count from fitted preprocessing.
# Input [B, features], output logits [B, 1], target float [B, 1].
# F.binary_cross_entropy_with_logits(logits, targets)

mnist = nn.Sequential(
    nn.Flatten(start_dim=1),
    MLP(784, [256, 128], 10, rngs=nj.Rngs(params=2)),
)
# Input [B, 1, 28, 28], output logits [B, 10], target integer [B].
```

Titanic preprocessing persists imputation, categorical vocabulary, feature order,
and scaling. A later mixed tabular model can combine embeddings and numeric
features with `jnp.concatenate`; it needs no new base class.

## CNNs: MNIST and cats versus dogs

```python
class DigitCNN(nn.Module):
    def __init__(self, *, rngs):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1, rngs=rngs)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, rngs=rngs)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.dropout = nn.Dropout(p=0.2)
        self.head = nn.Linear(64 * 7 * 7, 10, rngs=rngs)

    def forward(self, x, *, rngs=None):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.reshape(x.shape[0], -1)
        return self.head(self.dropout(x, rngs=rngs))

class PetCNN(nn.Module):
    def __init__(self, *, rngs):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1, rngs=rngs)
        self.norm1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1, rngs=rngs)
        self.norm2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(64, 2, rngs=rngs)

    def forward(self, x):
        x = self.pool(F.relu(self.norm1(self.conv1(x))))
        x = self.pool(F.relu(self.norm2(self.conv2(x))))
        return self.head(self.global_pool(x).reshape(x.shape[0], 64))
```

`DigitCNN`: `[B, 1, 28, 28] -> [B, 10]`. `PetCNN`:
`[B, 3, H, W] -> [B, 2]`, with spatial dimensions large enough for both pools.
BatchNorm owns affine parameters initialized without randomness and running-statistic
buffers. Training updates statistics; evaluation uses stored values. Input resize,
normalization, and augmentation belong to data transforms, not hidden model logic.
These are small teaching architectures, not promised competitive classifiers.

## RNNs: explicit sequence state

```python
class SequenceRegressor(nn.Module):
    def __init__(self, features, hidden, outputs, *, rngs):
        super().__init__()
        self.rnn = nn.RNN(features, hidden, num_layers=1,
                          nonlinearity="tanh", rngs=rngs)
        self.head = nn.Linear(hidden, outputs, rngs=rngs)

    def forward(self, x, state=None, *, lengths=None):
        sequence, next_state = self.rnn(x, state, lengths=lengths)
        return {"prediction": self.head(next_state[-1]),
                "state": next_state}
```

Input `[B, T, features]`; sequence output `[B, T, H]`; final hidden
`[layers, B, H]`. `state=None` means zeros, not a hidden persistent state.
For `lengths[B]`, the recurrent carry stops updating after each valid length;
padded outputs are zero. Final carry is the last valid carry. Zero-length samples
retain the input carry and require an explicit loss mask if excluded from training.
Without lengths, every timestep is valid.

`nn.RNNCell(input_size, hidden_size, ...)` supports custom loops with
`next_h = cell(x_t, h)`. The sequence module scans the cell with shared weights.
Streaming callers pass returned state to the next chunk. Truncated backpropagation
uses `nj.stop_gradient(state)`, which maps over a state pytree, at chunk boundaries.
Do not carry state between unrelated shuffled examples.

## LSTMs: memory is an explicit tuple

```python
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, classes, *, rngs):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 128, rngs=rngs)
        self.lstm = nn.LSTM(128, 256, num_layers=2, rngs=rngs)
        self.head = nn.Linear(256, classes, rngs=rngs)

    def forward(self, tokens, state=None, *, lengths=None):
        sequence, (h, c) = self.lstm(
            self.embedding(tokens), state, lengths=lengths,
        )
        return {"logits": self.head(h[-1]), "state": (h, c)}
```

Input tokens `[B, T]`; sequence `[B, T, 256]`; `h` and `c` both `[2, B, 256]`.
An `LSTMCell` maps `(x_t, (h, c)) -> (next_h, next_c)`; stacked sequence modules
return `(sequence, (h, c))`. Gate order is documented as input, forget, candidate,
output; initialize forget-gate bias to one and other biases to zero. No implicit
carry caching. Bidirectionality comes later with output width `2H`, hidden layout
`[layers * directions, B, H]`, and direction ordering specified in the layer docs.

## Transformers: attention, residuals, and masks

```python
class TransformerBlock(nn.Module):
    def __init__(self, width, heads, *, rngs):
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, rngs=rngs)
        self.norm2 = nn.LayerNorm(width)
        self.ff = nn.Sequential(
            nn.Linear(width, 4 * width, rngs=rngs),
            nn.GELU(),
            nn.Linear(4 * width, width, rngs=rngs),
        )

    def forward(self, x, *, mask=None):
        z = self.norm1(x)
        attended = self.attention(z, z, z, mask=mask)
        x = x + attended
        return x + self.ff(self.norm2(x))

class TinyLanguageModel(nn.Module):
    def __init__(self, vocab_size, max_length, width=128, depth=4,
                 heads=4, *, rngs):
        super().__init__()
        self.token = nn.Embedding(vocab_size, width, rngs=rngs)
        self.position = nn.Embedding(max_length, width, rngs=rngs)
        self.blocks = nn.ModuleList([
            TransformerBlock(width, heads, rngs=rngs) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, vocab_size, rngs=rngs)

    def forward(self, tokens, *, valid=None):
        length = tokens.shape[1]
        x = self.token(tokens) + self.position(jnp.arange(length))[None, :, :]
        mask = nn.attention.causal_mask(length)  # [1, 1, T, T]
        if valid is not None:
            mask = mask & nn.attention.padding_mask(valid)
        for block in self.blocks:
            x = block(x, mask=mask)
        return self.head(self.norm(x))
```

Attention accepts query `[B, Tq, D]`, key/value `[B, Tk, D]` in the initial
equal-width API; `D % heads == 0`. Default return is output `[B, Tq, D]`.
`need_weights=True` returns `(output, weights)` with weights `[B, heads, Tq, Tk]`;
that flag is static under compilation. Large attention weights are opt-in.

Boolean masks broadcast to `[B, heads, Tq, Tk]`; **True means allowed**. This is
an explicit convention, not a promise to match all PyTorch mask APIs.
`padding_mask(valid)` converts `[B, Tk]` to `[B, 1, 1, Tk]` for key masking.
Padded query locations are excluded separately from losses. Fully masked query
rows produce zero weights and zero attention output, including projection bias,
without NaNs. Masks combine by logical AND. Additive bias is a separately named
argument, never an overloaded boolean mask. Softmax normalization uses a stable
accumulation dtype.

The LM returns `[B, T, vocab_size]` logits. Next-token training supplies tokens
`[:, :-1]`, targets `[:, 1:]`, and the corresponding target-valid loss mask.
Position overflow raises an error. Padding validity and causal structure are
distinct; neither may be silently inferred from token values.

Encoder classification uses noncausal masks and valid-token pooling; a decoder
adds `MultiheadAttention(query, memory, memory)` for cross-attention. Vision
transformers reuse these blocks after patch embedding. Later KV-cache inference
must take and return explicit cache pytrees with capacity/position metadata;
it must not mutate a hidden Python list. Sampling requires an explicit RNG.

## Autoencoders: architecture and objective are separate

```python
class Autoencoder(nn.Module):
    def __init__(self, latent_size=16, *, rngs):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 256, rngs=rngs), nn.ReLU(),
            nn.Linear(256, latent_size, rngs=rngs),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, 256, rngs=rngs), nn.ReLU(),
            nn.Linear(256, 784, rngs=rngs), nn.Sigmoid(),
        )

    def encode(self, images):
        return self.encoder(images.reshape(images.shape[0], 784))

    def decode(self, latent):
        return self.decoder(latent).reshape(-1, 1, 28, 28)

    def forward(self, images):
        z = self.encode(images)
        return {"reconstruction": self.decode(z), "latent": z}

def reconstruction_objective(model, images):
    result = model(images)
    return F.mse_loss(result["reconstruction"], images)
```

Images are normalized to `[0, 1]`. Reconstruction and latent codes are named
outputs available to both losses and later visualization. A convolutional
autoencoder replaces the encoder/decoder with conv and transpose-conv modules;
explicit output-size checks prevent off-by-one reconstructions. A denoising
experiment uses corrupted inputs and clean targets without changing the model.

A future variational autoencoder returns `reconstruction_logits`, `mu`, and
`log_variance`. Reparameterization consumes a `sample` RNG, and the objective
declares reconstruction and KL reductions separately. This is an extension of
the same model and explicit loss contracts, not a new trainer hierarchy.

## GANs: two modules, two optimizers, deliberate gradient boundaries

```python
rngs = nj.Rngs(params=7)
generator = nn.Sequential(
    nn.Linear(64, 256, rngs=rngs), nn.ReLU(),
    nn.Linear(256, 784, rngs=rngs), nn.Tanh(),
)
discriminator = nn.Sequential(
    nn.Linear(784, 256, rngs=rngs), nn.ReLU(),
    nn.Linear(256, 1, rngs=rngs),
)
g_optimizer = optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
d_optimizer = optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))
sample_rngs = nj.Rngs(sample=8)

@nj.jit
def discriminator_step(generator, discriminator, optimizer, real, rngs):
    z = nj.random.normal(rngs.next("sample"), (real.shape[0], 64))
    fake = nj.stop_gradient(generator(z))

    def objective(candidate):
        real_logits = candidate(real)
        fake_logits = candidate(fake)
        return (
            F.binary_cross_entropy_with_logits(real_logits,
                                               jnp.ones_like(real_logits))
            + F.binary_cross_entropy_with_logits(fake_logits,
                                                 jnp.zeros_like(fake_logits))
        )

    loss, grads = nj.value_and_grad(objective)(discriminator)
    optimizer.step(grads)
    return {"d_loss": loss}

@nj.jit
def generator_step(generator, discriminator, optimizer, real, rngs):
    z = nj.random.normal(rngs.next("sample"), (real.shape[0], 64))

    def objective(candidate):
        logits = discriminator(candidate(z))
        return F.binary_cross_entropy_with_logits(logits, jnp.ones_like(logits))

    loss, grads = nj.value_and_grad(objective)(generator)
    optimizer.step(grads)
    return {"g_loss": loss}

for real_images, _ in train_loader:
    real = real_images.reshape(-1, 784)  # normalized to [-1, 1]
    generator.eval()
    discriminator.train()
    d_metrics = discriminator_step(
        generator, discriminator, d_optimizer, real, sample_rngs,
    )
    generator.train()
    discriminator.eval()
    g_metrics = generator_step(
        generator, discriminator, g_optimizer, real, sample_rngs,
    )
```

The discriminator phase detaches generated images. The generator phase
differentiates **through** the discriminator's input computation, while selecting
only generator parameters for gradients. Evaluation mode does not disable that
gradient path. Only the phase's optimizer steps; its counterpart's buffers are
held fixed by evaluation mode. These examples have no stochastic layers, but this
mode policy also defines what should happen when BatchNorm is later introduced.
An experiment may choose another policy explicitly.

Do not wrap the discriminator forward in `stop_gradient` during generator training:
that would remove the learning signal. Save both models, both optimizers, random
streams, and phase/update counters together. Fixed-noise galleries use a separate
saved evaluation seed, so visualization does not change training randomness.
GAN losses alone do not establish quality or rule out mode collapse.

## Extensibility beyond the cookbook

Custom modules can combine existing modules with `jax.numpy` operations, register
their own `Parameter` and `Buffer` objects, return structured outputs, and supply
serialization config. Graph inspection treats an unknown custom operation as an
opaque node with input/output metadata rather than inventing its internals.
Supported custom operations need valid JAX autodiff rules; truly new primitives
may require custom differentiation. Composition does not remove that requirement.
