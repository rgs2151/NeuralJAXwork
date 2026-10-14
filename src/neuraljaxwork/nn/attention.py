"""Batch-first attention. Boolean mask True means the key is allowed."""

import jax
import jax.numpy as jnp

from .layers import Linear
from .module import Module


def causal_mask(length):
    return jnp.tril(jnp.ones((length, length), dtype=bool))[None, None]


def padding_mask(valid):
    return jnp.asarray(valid, dtype=bool)[:, None, None, :]


class MultiheadAttention(Module):
    def __init__(self, embed_dim, num_heads, *, rngs):
        super().__init__()
        if num_heads < 1 or embed_dim % num_heads:
            raise ValueError("embed_dim must be divisible by positive num_heads")
        self.embed_dim, self.num_heads = embed_dim, num_heads
        self.q_proj = Linear(embed_dim, embed_dim, rngs=rngs)
        self.k_proj = Linear(embed_dim, embed_dim, rngs=rngs)
        self.v_proj = Linear(embed_dim, embed_dim, rngs=rngs)
        self.out_proj = Linear(embed_dim, embed_dim, rngs=rngs)

    def forward(self, query, key, value, *, mask=None, need_weights=False):
        if (
            query.ndim != 3
            or key.ndim != 3
            or value.shape != key.shape
            or query.shape[0] != key.shape[0]
        ):
            raise ValueError("Expected query [B,Tq,D], key/value [B,Tk,D]")
        batch, tq, _ = query.shape
        tk, heads = key.shape[1], self.num_heads
        width = self.embed_dim // heads

        def project(layer, x):
            return layer(x).reshape(batch, -1, heads, width).transpose(0, 2, 1, 3)

        q, k, v = (
            project(self.q_proj, query),
            project(self.k_proj, key),
            project(self.v_proj, value),
        )
        scores = jnp.einsum("bhqd,bhkd->bhqk", q, k).astype(jnp.float32) / jnp.sqrt(width)
        allowed = (
            jnp.ones((batch, heads, tq, tk), dtype=bool)
            if mask is None
            else jnp.broadcast_to(mask, scores.shape)
        )
        if allowed.dtype != jnp.bool_:
            raise ValueError("Attention masks must be boolean: True means allowed")
        nonempty = allowed.any(axis=-1, keepdims=True)
        masked = jnp.where(allowed, scores, -jnp.inf)
        safe = jnp.where(nonempty, masked, 0.0)
        weights = jnp.where(allowed, jax.nn.softmax(safe, axis=-1), 0.0).astype(v.dtype)
        context = jnp.einsum("bhqk,bhkd->bhqd", weights, v)
        output = self.out_proj(context.transpose(0, 2, 1, 3).reshape(batch, tq, self.embed_dim))
        output = jnp.where(nonempty.any(axis=1), output, 0.0)
        return (output, weights) if need_weights else output
