# 🐇NeuralJAXwork: GPU Accelerated Lightweight ML Framework from Scratch with JAX

[![Documentation](https://img.shields.io/badge/Complete-documentation-blue.svg)](https://rgs2151.github.io/NeuralJAXwork/) [![Documentation](https://img.shields.io/badge/api-reference-blue.svg)](https://rgs2151.github.io/NeuralJAXwork/_autosummary/NeuralJAXwork.html) ![Documentation](https://img.shields.io/badge/ONNX-supported-orange.svg) ![Documentation](https://img.shields.io/badge/GPU-supported-brightgreen.svg) ![Documentation](https://img.shields.io/badge/JIT-compiled-yellow.svg) [![Documentation](https://img.shields.io/badge/Habanero_HPC_Cluster-supported-green.svg)](https://confluence.columbia.edu/confluence/display/rcs/Habanero+HPC+Cluster+User+Documentation) ![Documentation](https://img.shields.io/badge/python-3.7_|_3.8_|_3.9_|_3.10-blue.svg)

NeuralJAXwork is a lightweight machine learning framework built from scratch using the JAX library, designed to accelerate model training on GPUs. It provides a high-level interface for building and training neural networks with ease, while also allowing for flexibility and customization through its low-level JAX API. With its efficient GPU acceleration and streamlined design, NeuralJAXwork is an ideal choice for researchers and practitioners looking to quickly prototype and experiment with new machine learning models. Its user-friendly interface and comprehensive documentation make it accessible to both novice and advanced users, while its performance and flexibility make it a powerful tool for a wide range of machine learning tasks.

---

**Colab Examples:**

| `MNIST`                                                                                                                                                                              | `XOR`                                                                                                                                                                                | `Titanic`                                                                                                                                                                            | `Cats & Dogs`                                                                                                                                                                        | `RegNet18`                                                             |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/googlecolab/colabtools/blob/master/notebooks/colab-github-demo.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/googlecolab/colabtools/blob/master/notebooks/colab-github-demo.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/googlecolab/colabtools/blob/master/notebooks/colab-github-demo.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/googlecolab/colabtools/blob/master/notebooks/colab-github-demo.ipynb) | ![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg) |

## Framework Skeleton

NeuralJAXwork is a well-structured and modular framework that is designed to be easily extensible with new classes and functions. The framework consists of four main modules: `activation.py`, `layer.py`, `loss.py`, and `model.py`, each of which plays a critical role in building and training neural networks. The `network.py` file serves as a bridge that brings all the modules together to create a complete and functional neural network.

The `activation.py` module contains code for various activation functions used in neural networks. The current implementation includes the popular `tanh` function, and the framework welcomes contributions of new activation functions, such as `sigmoid`, `relu`, and their derivatives. To ensure efficient computation, all activation functions can be `jit-compiled` for optimal performance on GPUs.

The `layer.py` module provides code for various layers used in neural networks. The current implementation includes the `Dense` layer, which is commonly used in fully connected neural networks. In addition to the `Dense` layer, the framework encourages contributions of other types of layers, such as convolutional layers, recurrent layers, and any other layer types required by specific models.

The `loss.py` module contains code for various loss functions used in neural networks. The current implementation includes popular loss functions, such as binary crossentropy, mean squared error (MSE), and mean absolute error (MAE). The framework also welcomes contributions of new loss functions, such as hinge loss, or any other loss functions that are required by specific models.

Finally, the `model.py` module provides a high-level interface for building and training neural networks using the classes and functions provided by the other modules. The `Model` class provides a simple and intuitive API that allows users to build, train, and evaluate neural networks with ease.

In addition to the four main modules, the framework provides comprehensive documentation and examples to help users get started and contribute to the project. The framework is designed to be lightweight and efficient, with a focus on GPU acceleration using the JAX library. It is an ideal choice for researchers and practitioners looking to quickly prototype and experiment with new machine learning models while having the flexibility to customize and extend the framework as needed.

### Models

#### Sqeuential

#### Functional

## Implementations

### Losses

| Loss Function                     | Implementation                                                                               | Prime                                                                                                               | Status |
| --------------------------------- | -------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- | ------ |
| Binary Crossentropy               | $BCE = -\frac{1}{n}\sum_{i=1}^{n} [y_{i}\log(\hat{y_{i}}) + (1-y_{i})\log(1-\hat{y_{i}})]$ | $\frac{\partial BCE}{\partial y_{pred}} = \frac{y_{pred}-y_{true}}{y_{pred}(1-y_{pred})}$                         | ✅     |
| Categorical Cross Entropy         | $CCE(y, \hat{y}) = - \sum_{i=1}^n y_i log(\hat{y_i})$                                      | $\frac{\partial CCE(y, \hat{y})}{\partial \hat{y_i}} = -\frac{y_i}{\hat{y_i}}$                                    | ✅     |
| Mean Squared Error (MSE)          | $MSE = \frac{1}{n} \sum_{i=1}^{n} (y_{true} - y_{pred})^2$                                 | $\frac{\partial MSE}{\partial y_{pred}} = \frac{2}{n} (y_{pred} - y_{true})$                                      | ✅     |
| Mean Absolute Error (MAE)         | $MAE = \frac{1}{n}\sum_{i=1}^{n}\|y_{true}-y_{pred}\|$                                     | $\frac{\partial MAE}{\partial y_{pred}} = \frac{1}{n} * sign(y_{pred} - y_{true})$                                | ✅     |
| Smooth Mean Absolute Error (sMAE) | $sMAE = \frac{1}{n} \sum_{i=1}^{n} (\sqrt{(y_{true} - y_{pred})^2 + \delta^2})$            | $\frac{\partial sMAE}{\partial y_{pred}} = \frac{y_{pred} - y_{true}}{\sqrt{(y_{pred} - y_{true})^2 + \delta^2}}$ | ✅     |

### Activations

| Activation Function | Implementation                                                          | Prime                                                                                     | Status |
| ------------------- | ----------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- | ------ |
| Sigmoid             | $tanh(x) = (e^x - e^-x) / (e^x + e^-x)$                               | $tanh'(x) = 1 - tanh(x)^2$                                                              | ✅     |
| Linear              | $f(x) = x$                                                            | $f'(x) = 1$                                                                             | ✅     |
| Softmax             | $Softmax(x_i) = \frac{e^{x_i}}{\sum_{j=1}^n e^{x_j}}$                 | $\frac{\partial Softmax(x_i)}{\partial x_j} = Softmax(x_i)(\delta_{ij} - Softmax(x_j))$ | ✅     |
| ReLU                | $ReLU(x) = max(0,x)$                                                  | $ReLU'(x) = \\begin{cases} 0& x <0 \\\\ 1& x \\geq 0 \\end{cases}$                      | ✅     |
| Leaky ReLU          | $LeakyReLU(x) = \begin{cases} x & x \geq 0 \\ ax & x < 0 \end{cases}$ | $LeakyReLU'(x) = \begin{cases} 1 & x \geq 0 \\ a & x < 0 \end{cases}$                   | ✅     |

### Layers

| Layer        | Implementation                      | Prime                                                                                                             | Status |
| ------------ | ----------------------------------- | ----------------------------------------------------------------------------------------------------------------- | ------ |
| Dense        | $\hat{Y}$ = $W \cdot X$ + $b$ | $\frac{\partial L}{\partial X} = \frac{\partial L}{\partial \hat{Y}} \cdot \frac{\partial \hat{Y}}{\partial X}$ | ✅     |
| Convolutions |                                     |                                                                                                                   | ✅     |
| LSTMs        |                                     |                                                                                                                   | 🚧     |

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
