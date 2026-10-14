# NeuralJAXwork

[Read the tutorial book](https://rgs2151.github.io/NeuralJAXwork/) ·
[Architecture tutorials](https://rgs2151.github.io/NeuralJAXwork/docs/tutorials.html)

<p align="center"><img src="logo/jax_logo_min.png" width="150" alt="NeuralJAXwork logo"></p>

[![Tests](https://github.com/rgs2151/NeuralJAXwork/actions/workflows/check_demos.yml/badge.svg)](https://github.com/rgs2151/NeuralJAXwork/actions/workflows/check_demos.yml)
[![Documentation](https://github.com/rgs2151/NeuralJAXwork/actions/workflows/doc_maker.yml/badge.svg)](https://rgs2151.github.io/NeuralJAXwork/)
[![Python](https://img.shields.io/badge/python-3.10%E2%80%933.13-blue)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![JAX](https://img.shields.io/badge/backend-JAX-009688)](https://github.com/jax-ml/jax)

NeuralJAXwork is a neural-network library with PyTorch-style module composition,
JAX automatic differentiation, and explicit training loops. Flax NNX manages
module state and transformations; Optax provides optimizer implementations.

Version **0.1.0 (alpha)**. Install from this repository; no PyPI release has been
published. The documentation includes API tutorials, reference material, and
locally trained examples with saved results. CI renders those results without
executing notebooks or training models.

## Install

Python 3.10–3.13; the tested local environment is Python 3.10 with JAX 0.5.3,
Flax 0.10.4, and Optax 0.2.4. The dependency bounds intentionally preserve this
NNX API generation. Use a fresh virtual environment.

```bash
python -m pip install "git+https://github.com/rgs2151/NeuralJAXwork.git"
```

That command installs the remote repository revision; local changes must first
be committed and pushed to be available there. To install this checkout now:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,examples]'
```

Plotting, pandas, and scikit-learn are optional. Add `.[notebooks]` for Jupyter,
or `.[docs]` for Sphinx. Core imports do not load notebook or plotting libraries.

## Define a model and train it

```python
import jax.numpy as jnp
import neuraljaxwork as nj
from neuraljaxwork import nn, optim
from neuraljaxwork.nn import functional as F

class Regressor(nn.Module):
    def __init__(self, *, rngs):
        super().__init__()
        self.hidden = nn.Linear(2, 8, rngs=rngs)
        self.head = nn.Linear(8, 1, rngs=rngs)

    def forward(self, x):
        return self.head(F.relu(self.hidden(x)))

model = Regressor(rngs=nj.Rngs(params=42))
optimizer = optim.Adam(model.parameters(), lr=0.01)

@nj.jit
def train_step(model, optimizer, x, y):
    loss, grads = nj.value_and_grad(lambda m: F.mse_loss(m(x), y))(model)
    optimizer.step(grads)
    return loss

x = jnp.array([[0., 1.], [1., 0.], [1., 1.]])
y = jnp.array([[1.], [1.], [2.]])
model.train()
for _ in range(50):
    loss = train_step(model, optimizer, x, y)
model.eval()
print(model(x))
```

There is no `loss.backward()` or accumulated `.grad` state. Use `nj.jit` around
functions that take modules, optimizers, and RNGs as explicit arguments; raw
`jax.jit` is for pure array functions. Model mode and gradient selection are
independent: evaluation mode still permits input gradients, as required by GANs.

## What's included

- Dense, convolutional and transpose-convolution layers, normalization, dropout,
  pooling, embeddings, RNN/LSTM cells and scanned sequence layers, masked attention.
- Reference MLP, digit CNN, RGB CNN, RNN/LSTM forecaster, LSTM classifier,
  transformer language model, dense/convolutional autoencoders, GAN components.
- SGD, Adam, AdamW, gradient clipping, deterministic batching, safe NPZ checkpoints,
  named parameters, JSON architecture hierarchy, local experiment records.
- Optional manual-optimization `Task`/`Trainer` and best-checkpoint callback.
- Tiny offline smoke tests and runnable dataset examples with plots.

All families share `nn.Module`; custom branches, residual paths, shared modules,
and structured outputs are ordinary Python composition.

## Examples

The [published results](https://rgs2151.github.io/NeuralJAXwork/docs/results.html)
come from local RTX 5090 training: 98.94% MNIST test accuracy, 81.22% Titanic
holdout accuracy, recurrent forecasting comparisons, MNIST reconstructions,
Tiny Shakespeare generation, and convolutional Fashion-MNIST samples. The
[API tutorials](https://rgs2151.github.io/NeuralJAXwork/docs/api_tutorials.html)
cover modules, optimization, data, checkpoints, and custom layers.

Run these from the repository root after installing `.[examples]`:

| Model | Example | Visual evidence |
| --- | --- | --- |
| Dense | `python -m examples.run titanic --download` | Held-out confusion matrix, ROC, calibration, subgroup report |
| CNN | `python -m examples.run cnn --download` | MNIST accuracy, confusion matrix, misclassified digits |
| RNN | `python -m examples.run rnn` | Sine next-value predictions and persistence baseline |
| LSTM | `python -m examples.run lstm` | Same forecasting task for comparison |
| Transformer | `python -m examples.run transformer` | Character loss/perplexity and sampled text |
| Autoencoder | `python -m examples.run autoencoder --download` | MNIST original/reconstruction pairs and latent codes |
| GAN | `python -m examples.run gan --download` | Fashion-MNIST fixed-noise sample grids and both losses |

Defaults use small subsets and two epochs for exploration, **not trained-model
quality claims**. Set `--epochs`, `--limit`, `--batch-size`, `--seed`, and `--output`.
`--limit` caps image/text training subsets and sets synthetic sequence count;
Titanic uses its full training split. `--smoke --epochs 1` uses tiny synthetic
inputs and no downloads. Synthetic smoke results are not dataset benchmarks.

The transformer includes an original repeated tiny corpus to work offline. For a
meaningful experiment pass `--text-file your-corpus.txt` (ASCII text you may use).
Its demo loss does not establish general language ability. GAN training can fail
or collapse; inspect fixed-noise samples rather than interpreting loss as quality.

Each run saves config, split/preprocessing metadata, architecture hierarchy,
metrics, checkpoints, plots, and a report under `runs/`. Existing run directories
are not silently overwritten. See [the example guide](docs/examples.md) and
[notebooks](notebooks/README.md).

## Fast verification

```bash
JAX_PLATFORMS=cpu python -m pytest
python -m build
python -m sphinx -W --keep-going -b html . _build/html
python scripts/check_docs.py _build/html
```

Tests use tiny batches, no downloads, and one or a few updates. They check actual
state/gradient behavior, not model convergence on full datasets. The notebook
and dataset runs are intentionally outside the default test suite. Documentation
CI renders saved notebook outputs and deploys successful main-branch builds to
GitHub Pages. Training is a separate, local GPU workflow.

## Documentation and status

- [API and semantics](docs/api.md)
- [Examples, datasets, and visualization](docs/examples.md)
- [Checkpoints and reproducibility](docs/state.md)
- [Development and support boundaries](docs/development.md)
- [Validation record](docs/validation.md)
- [Original design proposal](design/README.md) and [implementation status](design/IMPLEMENTATION.md)

Full operation-level architecture rendering, arbitrary model reconstruction from
JSON, parameter groups/freezing, bidirectional recurrence, KV caches, mixed
precision, and distributed training remain future work. The implementation-status
page records design differences explicitly.

The 2023 prototype is preserved in `legacy/` and excluded from the package.
The new lowercase import is `neuraljaxwork`; legacy checkpoints/API are incompatible.

MIT licensed. Dataset licenses and attribution are separate; see the example guide.
