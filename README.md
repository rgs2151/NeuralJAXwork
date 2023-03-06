# 🐇NeuralJAXwork: GPU Accelerated Lightweight ML Framework from Scratch with JAX

| **`Documentation`**                                                                              | `Colab Examples`                                                                                                                                                                     |
| -------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [![Documentation](https://img.shields.io/badge/api-reference-blue.svg)](https://rgs2151.github.io/NeuralJAXwork/) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/googlecolab/colabtools/blob/master/notebooks/colab-github-demo.ipynb) |

NeuralJAXwork is a lightweight machine learning framework built from scratch with the JAX library, designed to accelerate model training on GPUs. It provides a flexible and easy-to-use framework for building and training neural networks.

## Framework Skeleton

The framework is organized into four main modules: `activation.py`, `layer.py`, `loss.py`, and `model.py`, as well as a `network.py` file that brings all the modules together. The `__init__.py` file is used to make the modules importable as a package.

The framework skeleton exists and we encourage contributions to the corresponding directories by adding new classes. For example, you can add new activation functions such as `relu`, `sigmoid` and their derivatives using Jax for derivatives. Currently, only the `tanh` activation function exists.

### Losses

The losses directory contains code for various loss functions used in neural networks, including:

- Binary Crossentropy
- Hinge Loss
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- Smooth Mean Absolute Error (SMAE)

### Activation

The activations directory contains code for various activation functions used in neural networks, including:

- Sigmoid
- ReLU
- Linear
- Leaky ReLU
- Binary Step
- Softmax

### Layers

The layers directory contains code for various layers used in neural networks, including:

- Dense
- Convolutions
- LSTMs (if time permits)

## Usage

To use NeuralJAXwork, you can import the required modules and classes, and build your model as follows:

```python
from NeuralJAXwork.layers import Dense
from NeuralJAXwork.activations import ReLU
from NeuralJAXwork.losses import MSE
from NeuralJAXwork.model import Model

# Define the layers of the model
layers = [
    Dense(64),
    ReLU(),
    Dense(32),
    ReLU(),
    Dense(10)
]

# Define the loss function
loss_fn = MSE()

# Create the model
model = Model(layers=layers, loss_fn=loss_fn)

# Train the model
model.train(X_train, y_train, epochs=10, batch_size=32, learning_rate=0.001)
```
