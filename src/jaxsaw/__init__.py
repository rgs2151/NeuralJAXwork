"""JAXsaw: a lightweight, GPU-accelerated framework for neural networks in JAX."""

from . import nn, optim
from .core import (
    Rngs,
    apply,
    grad,
    jit,
    merge,
    random,
    split,
    stop_gradient,
    update,
    value_and_grad,
)

__version__ = "0.1.0"
from . import checkpoint, data, inspection, models, training

__all__ = [
    "Rngs",
    "jit",
    "grad",
    "value_and_grad",
    "split",
    "merge",
    "update",
    "apply",
    "stop_gradient",
    "random",
    "nn",
    "optim",
    "data",
    "checkpoint",
    "inspection",
    "models",
    "training",
]
