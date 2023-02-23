# Import JAX's version of NumPy
import jax.numpy as jnp 

# Import JAX's functions for compilation and differentiation
from jax import jit

# Import the Layer class
from njax import Layer

class Dense(Layer):
    """
    This class implements a dense layer.
    """

    # Decorate with jit to compile this function into XLA-optimized code
    @jit
    def __init__(self, input_size, output_size):
        """
        Initialize the weights and bias of the dense layer.

        Args:
            input_size: The number of inputs to the dense layer.
            output_size: The number of outputs of the dense layer.
        
        Returns:
            None
        """
        
        # Initialize the weights and bias
        self.weights = jnp.random.randn(output_size, input_size)
        self.bias = jnp.random.randn(output_size, 1)

    # Decorate with jit to compile this function into XLA-optimized code
    @jit
    def forward(self, input):
        """
        Compute the forward pass of the dense layer.

        Args:
            input: The input tensor.
        
        Returns:
            The dot product of the weights and the input plus the bias.
            
            > $\hat{Y}$ = $W \cdot X$ + $b$
        """

        # Save the input for the backward pass
        self.input = input

        # Compute the dot product of the weights and the input plus the bias
        return jnp.dot(self.weights, self.input) + self.bias

    # Decorate with jit to compile this function into XLA-optimized code
    @jit
    def backward(self, output_gradient, learning_rate):
        """
        Compute the backward pass of the dense layer.

        Args:
            output_gradient: The gradient of the output of the dense layer.
            learning_rate: The learning rate of the model.

        Returns:
            The gradient of the input of the dense layer.
            
            > $\frac{\partial L}{\partial X} = \frac{\partial L}{\partial \hat{Y}} \cdot \frac{\partial \hat{Y}}{\partial X}$
        """

        # Compute the gradient of the weights and bias
        weights_gradient = jnp.dot(output_gradient, self.input.T)

        # Compute the gradient of the input
        input_gradient = jnp.dot(self.weights.T, output_gradient)

        # Update the weights and bias using gradient descent and the learning rate
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * output_gradient
        return input_gradient
