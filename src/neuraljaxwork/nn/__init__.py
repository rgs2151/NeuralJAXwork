"""PyTorch-style module composition with JAX arrays and autodiff."""

from . import attention, functional
from .attention import MultiheadAttention
from .layers import (
    GELU,
    AdaptiveAvgPool2d,
    AvgPool2d,
    BatchNorm1d,
    BatchNorm2d,
    Conv1d,
    Conv2d,
    ConvTranspose2d,
    Dropout,
    Embedding,
    Flatten,
    LayerNorm,
    Linear,
    MaxPool2d,
    ReLU,
    Sigmoid,
    Tanh,
)
from .module import Buffer, Module, ModuleDict, ModuleList, Parameter, Sequential
from .recurrent import LSTM, RNN, LSTMCell, RNNCell

__all__ = [
    "Module",
    "ModuleList",
    "ModuleDict",
    "Sequential",
    "Parameter",
    "Buffer",
    "Linear",
    "Embedding",
    "ReLU",
    "GELU",
    "Tanh",
    "Sigmoid",
    "Flatten",
    "Dropout",
    "Conv2d",
    "MaxPool2d",
    "AdaptiveAvgPool2d",
    "LayerNorm",
    "BatchNorm2d",
    "functional",
    "RNNCell",
    "RNN",
    "LSTMCell",
    "LSTM",
    "MultiheadAttention",
    "attention",
    "Conv1d",
    "ConvTranspose2d",
    "AvgPool2d",
    "BatchNorm1d",
]
