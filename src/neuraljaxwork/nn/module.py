"""Composable modules with explicit parameters and non-trainable buffers."""

import jax
import numpy as np
from flax import nnx

Parameter = nnx.Param
Buffer = nnx.BatchStat


class ParameterSet:
    """Live parameter selection owned by a module (not an array snapshot)."""

    def __init__(self, module):
        self.module = module

    def __iter__(self):
        return (value for _, value in self.module.named_parameters())


class Module(nnx.Module):
    """Subclass and implement ``forward``; child modules are discovered by NNX."""

    def __init__(self):
        self.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def parameters(self):
        return ParameterSet(self)

    def named_parameters(self):
        for path, value in nnx.iter_graph(self):
            if isinstance(value, nnx.Param):
                yield ".".join(map(str, path)), value

    def named_buffers(self):
        for path, value in nnx.iter_graph(self):
            if isinstance(value, Buffer):
                yield ".".join(map(str, path)), value

    def buffers(self):
        return (v for _, v in self.named_buffers())

    def named_modules(self):
        for path, value in nnx.iter_graph(self):
            if isinstance(value, Module):
                yield ".".join(map(str, path)), value

    def train(self, mode=True):
        for _, module in self.named_modules():
            module.training = bool(mode)
        return self

    def eval(self):
        return self.train(False)

    def state_dict(self):
        """Copy canonical parameter/buffer arrays to host; no executable objects."""
        return {
            name: np.array(value.value, copy=True)
            for name, value in (*self.named_parameters(), *self.named_buffers())
        }

    def load_state_dict(self, state, strict=True):
        current = dict((*self.named_parameters(), *self.named_buffers()))
        if strict and current.keys() != state.keys():
            raise ValueError(f"State paths differ: {current.keys() ^ state.keys()}")
        pending = []
        for name in current.keys() & state.keys():
            array = np.asarray(state[name])
            old = current[name].value
            if array.shape != old.shape or array.dtype != old.dtype:
                raise ValueError(f"Shape/dtype mismatch at {name}")
            pending.append((current[name], jax.device_put(array)))
        for variable, array in pending:
            variable.value = array
        return self

    def to(self, *, device=None, dtype=None):
        for _, variable in (*self.named_parameters(), *self.named_buffers()):
            value = variable.value
            if dtype is not None and np.issubdtype(value.dtype, np.floating):
                value = value.astype(dtype)
            variable.value = jax.device_put(value, device)
        return self


class ModuleList(Module):
    """Registered ordered modules; iteration does not call them."""

    def __init__(self, modules=()):
        super().__init__()
        self.items = list(modules)

    def __iter__(self):
        return iter(self.items)

    def __getitem__(self, index):
        return self.items[index]

    def __len__(self):
        return len(self.items)


class ModuleDict(Module):
    def __init__(self, modules):
        super().__init__()
        self.items = dict(modules)

    def __getitem__(self, key):
        return self.items[key]


class Sequential(ModuleList):
    """Chain deterministic single-input layers. Use forward for RNG-aware branches."""

    def __init__(self, *modules):
        super().__init__(modules)

    def forward(self, x):
        for module in self:
            x = module(x)
        return x
