# Import JAX's version of NumPy
import jax.numpy as jnp 

# Imoprt JAX's random number generator
from jax import random

# Import the Layer class
from NeuralJAXwork.layer import Layer

# Set the JAX random number generator's seed
jax_key = random.PRNGKey(0)

class Dense(Layer):
    """
    This class implements a dense layer.
    
    """

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
        self.weights = random.uniform(jax_key, (output_size, input_size))
        self.bias = random.uniform(jax_key, (output_size, 1))

    def forward(self, input):
        """
        Compute the forward pass of the dense layer.

        Args:
            input: The input tensor.
        
        Returns:
            The dot product of the weights and the input plus the bias.
            
            >$\hat{Y}$ = $W \cdot X$ + $b$
        
        """

        # Save the input for the backward pass
        self.input = jnp.array(input)

        # Compute the dot product of the weights and the input plus the bias
        return jnp.dot(self.weights, self.input) + self.bias

    def backward(self, output_gradient, learning_rate):
        """
        Compute the backward pass of the dense layer.

        Args:
            output_gradient: The gradient of the output of the dense layer.
            learning_rate: The learning rate of the model.

        Returns:
            The gradient of the input of the dense layer.
            
            $\frac{\partial L}{\partial X} = \frac{\partial L}{\partial \hat{Y}} \cdot \frac{\partial \hat{Y}}{\partial X}$
        
        """

        # Compute the gradient of the weights and bias
        weights_gradient = jnp.dot(output_gradient, self.input.T)

        # Compute the gradient of the input
        input_gradient = jnp.dot(self.weights.T, output_gradient)

        # Update the weights and bias using gradient descent and the learning rate
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * output_gradient
        return input_gradient
