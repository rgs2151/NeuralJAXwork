# Import JAX's version of NumPy
import jax.numpy as jnp

# Import the parent Loss class
from njax import Loss

class BinaryCrossEntropy(Loss):
    """
    This class implements the Binary Cross Entropy loss function.
    """

    def __init__(self):
        
        def binary_cross_entropy(y_true, y_pred):
            """
            Implementation of the Binary Cross Entropy loss function
            
            Args:
                y_true: The true labels.
                y_pred: The predicted labels.

            Returns:
                The binary cross entropy.

                > $BCE = -\frac{1}{n} \sum_{i=1}^{n} (y_{true} \log(y_{pred}) + (1 - y_{true}) \log(1 - y_{pred}))$
            """

            # Return the binary cross entropy
            return jnp.mean(-y_true * jnp.log(y_pred) - (1 - y_true) * jnp.log(1 - y_pred))

        def binary_cross_entropy_prime(y_true, y_pred):
            """
            Implementation of the Binary Cross Entropy loss function's derivative

            Args:
                y_true: The true labels.
                y_pred: The predicted labels.

            Returns:
                The derivative of the binary cross entropy.

                > $\frac{\partial BCE}{\partial y_{pred}} = \frac{1}{n} \frac{y_{true}}{y_{pred}} - \frac{1 - y_{true}}{1 - y_{pred}}$
            """

            # Return the derivative of the binary cross entropy
            return ((1 - y_true) / (1 - y_pred) - y_true / y_pred) / jnp.size(y_true)
        
        # Call the parent class's constructor
        super().__init__(binary_cross_entropy, binary_cross_entropy_prime)
