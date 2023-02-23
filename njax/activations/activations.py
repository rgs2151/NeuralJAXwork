from njax.activation import Activation

# Import JAX's version of NumPy
import jax.numpy as jnp 

# Import JAX's functions for compilation and differentiation
from jax import jit, grad 

class Tanh(Activation):
    """
    The Tanh activation function.
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

                > 1 - jnp.tanh(x) ** 2
            """
            
            # Use grad to automatically differentiate tanh
            return grad(tanh)(x)
            
        super().__init__(tanh, tanh_prime)