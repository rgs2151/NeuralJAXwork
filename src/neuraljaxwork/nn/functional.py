"""Stable loss functions and stateless activations; classifier inputs are logits."""

import jax
import jax.numpy as jnp
import optax

relu = jax.nn.relu
gelu = jax.nn.gelu
sigmoid = jax.nn.sigmoid
tanh = jnp.tanh
softmax = jax.nn.softmax


def _reduce(value, reduction, mask=None):
    if mask is not None:
        if mask.shape != value.shape:
            raise ValueError("Mask must match unreduced loss shape")
        value = jnp.where(mask, value, 0)
    if reduction == "none":
        return value
    if reduction == "sum":
        return value.sum()
    if reduction == "mean":
        count = value.size if mask is None else mask.sum()
        return value.sum() / jnp.maximum(count, 1)
    raise ValueError("reduction must be 'none', 'sum', or 'mean'")


def _same_shape(a, b):
    if a.shape != b.shape:
        raise ValueError(f"Prediction/target shapes differ: {a.shape}, {b.shape}")


def cross_entropy(logits, targets, *, class_axis=-1, mask=None, reduction="mean"):
    logits = jnp.moveaxis(logits, class_axis, -1)
    if logits.shape[:-1] != targets.shape or not jnp.issubdtype(targets.dtype, jnp.integer):
        raise ValueError("Expected integer targets matching non-class dimensions")
    # Masked sentinel labels must never enter the gather.
    labels = targets if mask is None else jnp.where(mask, targets, 0)
    return _reduce(optax.softmax_cross_entropy_with_integer_labels(logits, labels), reduction, mask)


def binary_cross_entropy_with_logits(logits, targets, *, reduction="mean"):
    _same_shape(logits, targets)
    return _reduce(optax.sigmoid_binary_cross_entropy(logits, targets), reduction)


def mse_loss(pred, target, *, reduction="mean"):
    _same_shape(pred, target)
    return _reduce(jnp.square(pred - target), reduction)


def l1_loss(pred, target, *, reduction="mean"):
    _same_shape(pred, target)
    return _reduce(jnp.abs(pred - target), reduction)


def smooth_l1_loss(pred, target, *, beta=1.0, reduction="mean"):
    _same_shape(pred, target)
    if beta <= 0:
        raise ValueError("beta must be positive")
    d = jnp.abs(pred - target)
    return _reduce(jnp.where(d < beta, 0.5 * d**2 / beta, d - 0.5 * beta), reduction)
