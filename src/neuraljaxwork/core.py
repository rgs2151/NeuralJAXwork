"""State-aware JAX transformations backed by Flax NNX."""

import jax
from flax import nnx


class Rngs(nnx.Rngs):
    """Named, checkpointable random streams, e.g. ``Rngs(params=0, dropout=1)``."""

    def next(self, name):
        if name not in self:
            raise ValueError(f"Missing RNG stream {name!r}")
        return self[name]()


jit = nnx.jit
value_and_grad = nnx.value_and_grad
grad = nnx.grad
split = nnx.split
merge = nnx.merge
update = nnx.update
random = jax.random


def stop_gradient(tree):
    """Detach every array in a pytree, including recurrent state tuples."""
    return jax.tree.map(jax.lax.stop_gradient, tree)


def apply(graph, state, *args, **kwargs):
    """Pure application for deterministic modules; returns output and updated state."""
    model = nnx.merge(graph, state)
    output = model(*args, **kwargs)
    return output, nnx.state(model)
