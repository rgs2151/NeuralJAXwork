# Import JAX's version of NumPy
import jax.numpy as jnp

# Import the parent Loss class
from NeuralJAXwork.loss import Loss

class SMAE(Loss):
    """
    This class implements the Smooth Mean Absolute Error loss function.
    """

    def __init__(self, delta=1.0):
        self.delta = delta
        
        def smae(y_true, y_pred):
            """
            Implementation of the Smooth Mean Absolute Error loss function

            Args:
                y_true: The true labels.
                y_pred: The predicted labels.

            Returns:
                The smooth mean absolute error.

                > $SMAE = \frac{1}{n} \sum_{i=1}^{n} (\sqrt{(y_{true} - y_{pred})^2 + \delta^2})$
            """

            # Return the smooth mean absolute error
            return jnp.mean(jnp.sqrt(jnp.power(y_true - y_pred, 2) + jnp.power(self.delta, 2)))

        def smae_prime(y_true, y_pred):
            """
            Implementation of the Smooth Mean Absolute Error loss function's derivative

            Args:
                y_true: The true labels.
                y_pred: The predicted labels.

            Returns:
                The derivative of the smooth mean absolute error.

                > $\frac{\partial SMAE}{\partial y_{pred}} = \frac{y_{pred} - y_{true}}{\sqrt{(y_{pred} - y_{true})^2 + \delta^2}}$
            """

            # Return the derivative of the smooth mean absolute error
            return (y_pred - y_true) / jnp.sqrt(jnp.power(y_pred - y_true, 2) + jnp.power(self.delta, 2))
        
        # Call the parent class's constructor
        super().__init__(smae, smae_prime)
