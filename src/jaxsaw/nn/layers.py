"""Dense, convolutional, activation, and normalization layers."""

import math

import jax
import jax.numpy as jnp

from . import functional as F
from .module import Buffer, Module, Parameter


def _pair(value):
    return (value, value) if isinstance(value, int) else tuple(value)


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True, *, rngs):
        super().__init__()
        if min(in_features, out_features) < 1:
            raise ValueError("Feature counts must be positive")
        self.in_features, self.out_features = in_features, out_features
        bound = 1 / math.sqrt(in_features)
        self.weight = Parameter(
            jax.random.uniform(
                rngs.next("params"), (out_features, in_features), minval=-bound, maxval=bound
            )
        )
        self.bias = Parameter(jnp.zeros(out_features)) if bias else None

    def forward(self, x):
        if x.shape[-1] != self.in_features:
            raise ValueError(f"Expected last dimension {self.in_features}, got {x.shape}")
        result = x @ self.weight.value.T
        return result if self.bias is None else result + self.bias.value


class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, *, rngs):
        super().__init__()
        self.weight = Parameter(
            jax.random.normal(rngs.next("params"), (num_embeddings, embedding_dim)) * 0.02
        )

    def forward(self, indices):
        if not jnp.issubdtype(indices.dtype, jnp.integer):
            raise ValueError("Embedding indices must be integers")
        return self.weight.value[indices]


class ReLU(Module):
    def forward(self, x):
        return F.relu(x)


class GELU(Module):
    def forward(self, x):
        return F.gelu(x)


class Tanh(Module):
    def forward(self, x):
        return jnp.tanh(x)


class Sigmoid(Module):
    def forward(self, x):
        return F.sigmoid(x)


class Flatten(Module):
    def __init__(self, start_dim=1):
        super().__init__()
        self.start_dim = start_dim

    def forward(self, x):
        return x.reshape(*x.shape[: self.start_dim], -1)


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        if not 0 <= p < 1:
            raise ValueError("Dropout probability must be in [0, 1)")
        self.p = p

    def forward(self, x, *, rngs=None):
        if not self.training or self.p == 0:
            return x
        if rngs is None:
            raise ValueError("Training dropout requires rngs with a dropout stream")
        keep = jax.random.bernoulli(rngs.next("dropout"), 1 - self.p, x.shape)
        return jnp.where(keep, x / (1 - self.p), 0)


class Conv2d(Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        groups=1,
        bias=True,
        *,
        rngs,
    ):
        super().__init__()
        if in_channels % groups or out_channels % groups:
            raise ValueError("Channels must be divisible by groups")
        self.in_channels, self.out_channels = in_channels, out_channels
        self.kernel_size, self.stride = _pair(kernel_size), _pair(stride)
        self.padding, self.groups = padding, groups
        fan_in = (in_channels // groups) * math.prod(self.kernel_size)
        self.weight = Parameter(
            jax.random.uniform(
                rngs.next("params"),
                (out_channels, in_channels // groups, *self.kernel_size),
                minval=-1 / math.sqrt(fan_in),
                maxval=1 / math.sqrt(fan_in),
            )
        )
        self.bias = Parameter(jnp.zeros(out_channels)) if bias else None

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != self.in_channels:
            raise ValueError("Conv2d expects [batch, in_channels, height, width]")
        padding = (
            self.padding.upper()
            if isinstance(self.padding, str)
            else tuple((p, p) for p in _pair(self.padding))
        )
        y = jax.lax.conv_general_dilated(
            x,
            self.weight.value,
            self.stride,
            padding,
            dimension_numbers=("NCHW", "OIHW", "NCHW"),
            feature_group_count=self.groups,
        )
        return y if self.bias is None else y + self.bias.value[None, :, None, None]


class MaxPool2d(Module):
    def __init__(self, kernel_size, stride=None):
        super().__init__()
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride) if stride is not None else self.kernel_size

    def forward(self, x):
        return jax.lax.reduce_window(
            x, -jnp.inf, jax.lax.max, (1, 1, *self.kernel_size), (1, 1, *self.stride), "VALID"
        )


class AdaptiveAvgPool2d(Module):
    def __init__(self, output_size=(1, 1)):
        super().__init__()
        if _pair(output_size) != (1, 1):
            raise ValueError("Initial implementation supports global pooling only: (1, 1)")

    def forward(self, x):
        return x.mean(axis=(-2, -1), keepdims=True)


class LayerNorm(Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        self.shape = (
            (normalized_shape,) if isinstance(normalized_shape, int) else tuple(normalized_shape)
        )
        self.eps = eps
        self.weight, self.bias = Parameter(jnp.ones(self.shape)), Parameter(jnp.zeros(self.shape))

    def forward(self, x):
        if x.shape[-len(self.shape) :] != self.shape:
            raise ValueError("LayerNorm trailing shape mismatch")
        axes = tuple(range(x.ndim - len(self.shape), x.ndim))
        mean = x.mean(axes, keepdims=True)
        variance = jnp.mean(jnp.square(x - mean), axes, keepdims=True)
        return (x - mean) * jax.lax.rsqrt(variance + self.eps) * self.weight.value + self.bias.value


class BatchNorm2d(Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.eps, self.momentum = eps, momentum
        self.weight, self.bias = (
            Parameter(jnp.ones(num_features)),
            Parameter(jnp.zeros(num_features)),
        )
        self.running_mean, self.running_var = (
            Buffer(jnp.zeros(num_features)),
            Buffer(jnp.ones(num_features)),
        )

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != self.weight.value.size:
            raise ValueError("BatchNorm2d expects NCHW with matching channels")
        if self.training:
            mean, var = x.mean((0, 2, 3)), x.var((0, 2, 3))
            self.running_mean.value = (
                1 - self.momentum
            ) * self.running_mean.value + self.momentum * mean
            self.running_var.value = (
                1 - self.momentum
            ) * self.running_var.value + self.momentum * var
        else:
            mean, var = self.running_mean.value, self.running_var.value
        shape = (1, -1, 1, 1)
        return (x - mean.reshape(shape)) * jax.lax.rsqrt(
            var.reshape(shape) + self.eps
        ) * self.weight.value.reshape(shape) + self.bias.value.reshape(shape)


class Conv1d(Module):
    """NCL convolution implemented through the same checked Conv2d primitive."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        groups=1,
        bias=True,
        *,
        rngs,
    ):
        super().__init__()
        self.conv = Conv2d(
            in_channels,
            out_channels,
            (1, kernel_size),
            stride=(1, stride),
            padding=(0, padding),
            groups=groups,
            bias=bias,
            rngs=rngs,
        )

    def forward(self, x):
        if x.ndim != 3:
            raise ValueError("Conv1d expects [batch, channels, length]")
        return self.conv(x[:, :, None, :])[:, :, 0, :]


class ConvTranspose2d(Module):
    """Transposed NCHW convolution, with explicit integer padding/output_padding."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        output_padding=0,
        bias=True,
        *,
        rngs,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size, self.stride = _pair(kernel_size), _pair(stride)
        self.padding, self.output_padding = _pair(padding), _pair(output_padding)
        if any(p < 0 or p >= s for p, s in zip(self.output_padding, self.stride, strict=True)):
            raise ValueError("output_padding must be nonnegative and smaller than stride")
        fan_in = in_channels * math.prod(self.kernel_size)
        self.weight = Parameter(
            jax.random.uniform(
                rngs.next("params"),
                (out_channels, in_channels, *self.kernel_size),
                minval=-1 / math.sqrt(fan_in),
                maxval=1 / math.sqrt(fan_in),
            )
        )
        self.bias = Parameter(jnp.zeros(out_channels)) if bias else None

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != self.in_channels:
            raise ValueError("ConvTranspose2d expects NCHW with matching channels")
        padding = tuple(
            (k - 1 - p, k - 1 - p + op)
            for k, p, op in zip(self.kernel_size, self.padding, self.output_padding, strict=True)
        )
        output = jax.lax.conv_general_dilated(
            x,
            self.weight.value[:, :, ::-1, ::-1],
            (1, 1),
            padding,
            lhs_dilation=self.stride,
            dimension_numbers=("NCHW", "OIHW", "NCHW"),
        )
        return output if self.bias is None else output + self.bias.value[None, :, None, None]


class AvgPool2d(MaxPool2d):
    def forward(self, x):
        return jax.lax.reduce_window(
            x, 0.0, jax.lax.add, (1, 1, *self.kernel_size), (1, 1, *self.stride), "VALID"
        ) / math.prod(self.kernel_size)


class BatchNorm1d(BatchNorm2d):
    def forward(self, x):
        if x.ndim == 2:
            return super().forward(x[:, :, None, None])[:, :, 0, 0]
        if x.ndim == 3:
            return super().forward(x[:, :, None, :])[:, :, 0, :]
        raise ValueError("BatchNorm1d expects [B,C] or [B,C,L]")
