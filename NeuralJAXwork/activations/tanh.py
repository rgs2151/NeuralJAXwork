# Import JAX's version of NumPy
import jax.numpy as jnp 

# Import JAX's functions for compilation and differentiation
from jax import jit

# Import the Activation Layer class
from NeuralJAXwork.activation import Activation

class Tanh(Activation):
    """
    This class implements the Tanh activation function.
    """
    def __init__(self):

        # Decorate with jit to compile this function into XLA-optimized code
        @jit
        def tanh(x):
            """
            Compute the Tanh activation of the input.

            Args:
                x: Input tensor.

            Returns:
                The Tanh activation of the input tensor.

                tanh(x) = (e^x - e^-x) / (e^x + e^-x)
            """

            # Use JAX's tanh function
            return jnp.tanh(x)

        # Decorate with jit to compile this function into XLA-optimized code
        @jit
        def tanh_prime(x):
            """
            Compute the gradient of the Tanh activation function using Autograd.

            Args:
                x: Input tensor.

            Returns:
                The gradient of the Tanh activation function.

                tanh'(x) = 1 - tanh(x)^2
            """
            
            # Differentiate tanh using JAX's numpy
            return 1 - jnp.tanh(x) ** 2
            
        super().__init__(tanh, tanh_prime)