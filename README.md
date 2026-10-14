<p align="center"><img src="logo/neuraljaxlib.svg" width="150" alt="NeuralJAXlib logo"></p>

# NeuralJAXlib

[![Tests](https://github.com/rgs2151/NeuralJAXlib/actions/workflows/check_demos.yml/badge.svg)](https://github.com/rgs2151/NeuralJAXlib/actions/workflows/check_demos.yml)
[![Documentation](https://github.com/rgs2151/NeuralJAXlib/actions/workflows/doc_maker.yml/badge.svg)](https://rgs2151.github.io/NeuralJAXlib/)
[![Python](https://img.shields.io/badge/python-3.10%E2%80%933.13-blue)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

NeuralJAXlib is a lightweight, GPU-accelerated framework for building and exploring
neural networks in JAX. Its modular building blocks support architectures ranging
from dense networks and CNNs to transformers, autoencoders, and GANs.

The library supports model development, training, and evaluation, with documented
examples and visualizations for learning and experimentation.

## Gallery

[CNN, RNN, Autoencoder, GAN, and Transformer animated gallery](gallery/assets/composite.html)

[Documentation](https://rgs2151.github.io/NeuralJAXlib/) ·
[Architecture tutorials](https://rgs2151.github.io/NeuralJAXlib/docs/learning.html) ·
[API tutorials](https://rgs2151.github.io/NeuralJAXlib/docs/api_tutorials.html)

## Install

Requires Python 3.10–3.13. Runtime dependencies include JAX, Flax NNX, and Optax;
version bounds are specified in `pyproject.toml`.

```bash
python -m pip install "git+https://github.com/rgs2151/NeuralJAXlib.git"
```

For an editable installation with development and example dependencies:

```bash
git clone https://github.com/rgs2151/NeuralJAXlib.git
cd NeuralJAXlib
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,examples]'
```

Plotting, pandas, and scikit-learn are optional. Add `.[notebooks]` for Jupyter,
or `.[docs]` for Sphinx. Core imports do not load notebook or plotting libraries.

## Define a model and train it

```python
import jax.numpy as jnp
import neuraljaxlib as nj
from neuraljaxlib import nn, optim
from neuraljaxlib.nn import functional as F

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

## Features

- Dense, convolutional and transpose-convolution layers, normalization, dropout,
  pooling, embeddings, RNN/LSTM cells and scanned sequence layers, masked attention.
- Reference MLP, digit CNN, RGB CNN, RNN/LSTM forecaster, LSTM classifier,
  transformer language model, dense/convolutional autoencoders, GAN components.
- SGD, Adam, AdamW, gradient clipping, deterministic batching, safe NPZ checkpoints,
  named parameters, JSON architecture hierarchy, local experiment records.
- Optional manual-optimization `Task`/`Trainer` and best-checkpoint callback.
- Dataset examples with evaluation metrics and plots.

All families share `nn.Module`; custom branches, residual paths, shared modules,
and structured outputs are ordinary Python composition.

## Examples

The [architecture tutorials](https://rgs2151.github.io/NeuralJAXlib/docs/learning.html)
cover tabular classification, image classification, sequence forecasting, language
modeling, reconstruction, and image generation. Each includes training code,
evaluation, and saved figures. The
[API tutorials](https://rgs2151.github.io/NeuralJAXlib/docs/api_tutorials.html)
cover modules, optimization, data, checkpoints, and custom layers.

Run these from the repository root after installing `.[examples]`:

| Model | Command | Outputs |
| --- | --- | --- |
| Dense | `python -m examples.run titanic --download` | Held-out confusion matrix, ROC, calibration, subgroup report |
| CNN | `python -m examples.run cnn --download` | MNIST accuracy, correct predictions, confusion matrix, and error analysis |
| RNN | `python -m examples.run rnn --download` | Next-day New York temperature curves and seasonal baselines |
| LSTM | `python -m examples.run lstm --download` | S&P 500 volatility forecasts and historical baselines |
| Transformer | `python -m examples.run transformer` | Character loss/perplexity and sampled text |
| Autoencoder | `python -m examples.run autoencoder --download` | MNIST original/reconstruction pairs and latent codes |
| GAN | `python -m examples.run gan --download` | Fashion-MNIST fixed-noise sample grids and both losses |

Defaults use small subsets and two epochs. Configure a run with
`--epochs`, `--limit`, `--batch-size`, `--seed`, and `--output`.
`--limit` caps image/text training subsets and forecasting windows;
Titanic uses its full training split. `--smoke --epochs 1` uses tiny synthetic
inputs without downloads.

For language modeling, supply an ASCII corpus with `--text-file your-corpus.txt`;
the default repeated text is for checking the pipeline. GAN evaluation uses
fixed-noise sample grids alongside training losses.

Each run saves config, split/preprocessing metadata, architecture hierarchy,
metrics, checkpoints, plots, and a report under `runs/`. Each run requires a new
output directory. See [the example guide](docs/examples.md) and
[notebooks](notebooks/README.md).

## Development

```bash
python -m pip install -e '.[dev,docs]'
JAX_PLATFORMS=cpu python -m pytest
python -m build
python -m sphinx -W --keep-going -b html . _build/html
python scripts/check_docs.py _build/html
```

Tests check numerical and state behavior on small arrays without dataset downloads.
Documentation builds render saved notebook outputs without executing cells.
See the [development guide](maintainers/development.md) for contribution and
validation procedures.

## Documentation

- [API and semantics](docs/api.md)
- [Examples, datasets, and visualization](docs/examples.md)
- [Checkpoints and reproducibility](docs/state.md)
- [Compatibility and limitations](docs/compatibility.md)
- [Contributor and maintainer records](maintainers/README.md)
- [Original design proposal](design/README.md) and [implementation status](design/IMPLEMENTATION.md)

The 2023 prototype is preserved in `legacy/` and excluded from the package.
Its API and checkpoints are incompatible with `neuraljaxlib`.

MIT licensed. Dataset licenses and attribution are separate; see the example guide.
