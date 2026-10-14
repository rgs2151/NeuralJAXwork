""" NeuralJAXwork is a GPU-accelerated, lightweight machine learning framework built from scratch using the JAX library in Python. 
It provides a high-level interface for building and training neural networks with ease, while also allowing for flexibility and customization through its low-level JAX API. 
With its efficient GPU acceleration and streamlined design, NeuralJAXwork is an ideal choice for researchers and practitioners looking to quickly prototype and experiment with new machine learning models. 
Its user-friendly interface and comprehensive documentation make it accessible to both novice and advanced users, while its performance and flexibility make it a powerful tool for a wide range of machine learning tasks.
"""

# Import the Layer class
from .layer import Layer

# Import the Activation Layer class
from .activation import Activation

# Import the Loss class
from .loss import Loss

# Import the Model class
from .model import Model

# Import errors
from .errors import Errors