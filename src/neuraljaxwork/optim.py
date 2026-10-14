"""Optimizers with explicit gradients and live module parameter ownership."""

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from .nn.module import ParameterSet


class Optimizer(nnx.Object):
    def __init__(self, parameters, transformation):
        if not isinstance(parameters, ParameterSet):
            raise TypeError("Pass model.parameters(); parameter groups are not supported yet")
        self.inner = nnx.Optimizer(parameters.module, transformation)

    @property
    def step_count(self):
        return self.inner.step.value

    def step(self, grads):
        """Apply one update. Gradients must come from the same module structure."""
        self.inner.update(grads)


class SGD(Optimizer):
    def __init__(self, parameters, lr=0.01, momentum=0.0):
        super().__init__(parameters, optax.sgd(lr, momentum=momentum))
        self.config = {"type": "SGD", "lr": lr, "momentum": momentum}


class Adam(Optimizer):
    def __init__(self, parameters, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        super().__init__(parameters, optax.adam(lr, b1=betas[0], b2=betas[1], eps=eps))
        self.config = {"type": "Adam", "lr": lr, "betas": list(betas), "eps": eps}


class AdamW(Optimizer):
    def __init__(self, parameters, lr=1e-3, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8):
        super().__init__(
            parameters,
            optax.adamw(lr, b1=betas[0], b2=betas[1], eps=eps, weight_decay=weight_decay),
        )
        self.config = {
            "type": "AdamW",
            "lr": lr,
            "betas": list(betas),
            "eps": eps,
            "weight_decay": weight_decay,
        }


def clip_grad_norm(grads, max_norm):
    if max_norm <= 0:
        raise ValueError("max_norm must be positive")
    norm = optax.global_norm(grads)
    scale = jnp.minimum(1.0, max_norm / jnp.maximum(norm, 1e-12))
    return jax.tree.map(lambda x: x * scale, grads), norm
