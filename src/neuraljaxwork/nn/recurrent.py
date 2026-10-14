"""Batch-first recurrent layers; carries are explicit and sequence loops use scan."""

import jax
import jax.numpy as jnp

from .layers import Linear
from .module import Module, ModuleList


class RNNCell(Module):
    def __init__(self, input_size, hidden_size, nonlinearity="tanh", *, rngs):
        super().__init__()
        if nonlinearity not in ("tanh", "relu"):
            raise ValueError("nonlinearity must be tanh or relu")
        self.input = Linear(input_size, hidden_size, rngs=rngs)
        self.hidden = Linear(hidden_size, hidden_size, bias=False, rngs=rngs)
        self.nonlinearity = nonlinearity

    def forward(self, x, h):
        z = self.input(x) + self.hidden(h)
        return jnp.tanh(z) if self.nonlinearity == "tanh" else jax.nn.relu(z)


class LSTMCell(Module):
    def __init__(self, input_size, hidden_size, *, rngs):
        super().__init__()
        self.input = Linear(input_size, 4 * hidden_size, rngs=rngs)
        self.hidden = Linear(hidden_size, 4 * hidden_size, bias=False, rngs=rngs)
        self.input.bias.value = self.input.bias.value.at[hidden_size : 2 * hidden_size].set(1.0)

    def forward(self, x, state):
        h, c = state
        i, f, g, o = jnp.split(self.input(x) + self.hidden(h), 4, axis=-1)
        c = jax.nn.sigmoid(f) * c + jax.nn.sigmoid(i) * jnp.tanh(g)
        return jax.nn.sigmoid(o) * jnp.tanh(c), c


class RNN(Module):
    cell_type = RNNCell

    def __init__(self, input_size, hidden_size, num_layers=1, nonlinearity="tanh", *, rngs):
        super().__init__()
        if min(input_size, hidden_size, num_layers) < 1:
            raise ValueError("Sizes and layer count must be positive")
        self.hidden_size, self.num_layers = hidden_size, num_layers
        kwargs = {"nonlinearity": nonlinearity} if self.cell_type is RNNCell else {}
        self.cells = ModuleList(
            [
                self.cell_type(
                    input_size if i == 0 else hidden_size, hidden_size, rngs=rngs, **kwargs
                )
                for i in range(num_layers)
            ]
        )

    def forward(self, x, state=None, *, lengths=None):
        if x.ndim != 3:
            raise ValueError("Recurrent input must be [batch, time, features]")
        batch, time, _ = x.shape
        shape = (self.num_layers, batch, self.hidden_size)
        is_lstm = self.cell_type is LSTMCell
        if state is None:
            zero = jnp.zeros(shape, dtype=x.dtype)
            state = (zero, zero) if is_lstm else zero
        for leaf in jax.tree.leaves(state):
            if leaf.shape != shape:
                raise ValueError(f"Expected recurrent state shape {shape}")
        if is_lstm and (not isinstance(state, tuple) or len(state) != 2):
            raise ValueError("LSTM state must be an (h, c) tuple")
        lengths = jnp.full((batch,), time) if lengths is None else jnp.asarray(lengths)
        if lengths.shape != (batch,):
            raise ValueError("lengths must have shape [batch]")
        final = []
        for layer, cell in enumerate(self.cells):
            initial = (state[0][layer], state[1][layer]) if is_lstm else state[layer]

            def step(carry, inputs, cell=cell):
                index, value = inputs
                active = (index < lengths)[:, None]
                candidate = cell(value, carry)
                carry = jax.tree.map(lambda new, old: jnp.where(active, new, old), candidate, carry)
                h = carry[0] if is_lstm else carry
                return carry, jnp.where(active, h, 0)

            carry, sequence = jax.lax.scan(step, initial, (jnp.arange(time), x.swapaxes(0, 1)))
            final.append(carry)
            x = sequence.swapaxes(0, 1)
        result = (
            (jnp.stack([s[0] for s in final]), jnp.stack([s[1] for s in final]))
            if is_lstm
            else jnp.stack(final)
        )
        return x, result


class LSTM(RNN):
    cell_type = LSTMCell

    def __init__(self, input_size, hidden_size, num_layers=1, *, rngs):
        super().__init__(input_size, hidden_size, num_layers, rngs=rngs)
