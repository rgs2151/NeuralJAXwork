# Import JAX's version of NumPy
import jax.numpy as jnp

# Import the parent Loss class
from NeuralJAXwork.loss import Loss

class MAE(Loss):
    """
    This class implements the Mean Absolute Error loss function.
    """

    def __init__(self):
        
        def mae(y_true, y_pred):
            """
            Implementation of the Mean Absolute Error loss function

            Args:
                y_true: The true labels.
                y_pred: The predicted labels.

            Returns:
                The mean absolute error.

                > $MAE = \frac{1}{n} \\sum_{i=1}^{n} |y_{true} - y_{pred}|$
            """

            # Return the mean absolute error
            return jnp.mean(jnp.abs(y_true - y_pred))

        def mae_prime(y_true, y_pred):
            """
            Implementation of the Mean Absolute Error loss function's derivative

            Args:
                y_true: The true labels.
                y_pred: The predicted labels.

            Returns:
                The derivative of the mean absolute error.

                > $\frac{\partial MAE}{\partial y_{pred}} = \frac{1}{n} * sign(y_{pred} - y_{true})$
            """

            # Return the derivative of the mean absolute error
            return jnp.sign(y_pred - y_true) / jnp.size(y_true)
        
        # Call the parent class's constructor
        super().__init__(mae, mae_prime)