# 🐇NeuralJAXwork: GPU Accelerated Lightweight ML Framework from Scratch with JAX

| **`Documentation`**                                                                                  | `Colab Examples`                                                                                                                                                                     |
| ------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [![Documentation](https://img.shields.io/badge/api-reference-blue.svg)](https://rgs2151.github.io/NeuralJAXwork/) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/googlecolab/colabtools/blob/master/notebooks/colab-github-demo.ipynb) |

NeuralJAXwork is a lightweight machine learning framework built from scratch using the JAX library, designed to accelerate model training on GPUs. It provides a high-level interface for building and training neural networks with ease, while also allowing for flexibility and customization through its low-level JAX API. With its efficient GPU acceleration and streamlined design, NeuralJAXwork is an ideal choice for researchers and practitioners looking to quickly prototype and experiment with new machine learning models. Its user-friendly interface and comprehensive documentation make it accessible to both novice and advanced users, while its performance and flexibility make it a powerful tool for a wide range of machine learning tasks.

## Framework Skeleton

The framework is organized into four main modules: `activation.py`, `layer.py`, `loss.py`, and `model.py`, as well as a `network.py` file that brings all the modules together.

The framework skeleton exists and we encourage contributions to the corresponding directories by adding new classes. For example, you can add new activation functions such as `relu`, `sigmoid` and their derivatives that can be `jit-compiled`.

### Losses

| Loss Function                     | Implementation File | Status          |
| --------------------------------- | ------------------- | --------------- |
| Binary Crossentropy               |                     | Implemented ✅     |
| Hinge Loss                        |                     | Not Implemented |
| Mean Squared Error (MSE)          |                     | Implemented ✅     |
| Mean Absolute Error (MAE)         |                     | Implemented ✅     |
| Smooth Mean Absolute Error (SMAE) |                     | Implemented ✅     |

### Activation

| Activation Function | Implementation File | Status      |
| ------------------- | ------------------- | ----------- |
| Sigmoid             |                     | Implemented ✅ |
| ReLU                |                     | Implemented ✅ |
| Linear              |                     | Implemented ✅ |
| Leaky ReLU          |                     | Implemented ✅ |
| Binary Step         |                     | Implemented ✅ |
| Softmax             |                     | Implemented ✅ |

### Layers

| Layer        | Implementation                | Status          |
| ------------ | ----------------------------- | --------------- |
| Dense        | NeuralJAXwork/layers/dense.py | Implemented ✅     |
| Convolutions |                               | Not Implemented |
| LSTMs        |                               | Not Implemented |

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
