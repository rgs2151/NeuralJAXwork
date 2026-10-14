# Getting started

NeuralJAXwork combines familiar module composition with JAX's explicit gradients
and state-aware compilation. This alpha installs from the repository; it is not
yet published to PyPI.

## Install a checkout

```bash
git clone https://github.com/rgs2151/NeuralJAXwork.git
cd NeuralJAXwork
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,examples]'
```

Use Python 3.10–3.13. The supported dependency range is pinned around JAX 0.5.3
and Flax NNX 0.10.4; see the [development guide](development.md) for limits.

## Your first explicit training step

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
def step(model, optimizer, x, y):
    loss, grads = nj.value_and_grad(lambda m: F.mse_loss(m(x), y))(model)
    optimizer.step(grads)
    return loss

x = jnp.array([[0., 1.], [1., 0.], [1., 1.]])
y = jnp.array([[1.], [1.], [2.]])
model.train()
for _ in range(50):
    loss = step(model, optimizer, x, y)
print(model.eval()(x))
```

The module owns weights and buffers. The optimizer owns update state. Gradients
are returned explicitly: there is no `loss.backward()` or persistent `.grad`.
`nj.jit` preserves the relationships among modules, optimizers, and RNG objects.

## Run a tutorial

```bash
python -m examples.run cnn --smoke --epochs 1
python -m examples.run cnn --download --epochs 10 --limit 10000
```

The first command verifies the pipeline on tiny synthetic data. The second runs
the MNIST experiment. Both write fresh run directories containing metrics,
checkpoints, and plots. Start with the [tutorial chapters](tutorials.rst), then
use the [API guide](api.md) to build your own model.

For interactive editing, install `.[notebooks]` and run `jupyter lab`. The same
notebooks are executed and rendered into this book by the documentation build.
