# Import JAX's version of NumPy
import jax.numpy as jnp

from njax import Loss


class MSE(Loss):
    '''
    This class implements the Mean Squared Error loss function.
    '''

    def __init__(self):
        
        def mse(y_true, y_pred):
            '''
            Implementation of the Mean Squared Error loss function

            Args:
                y_true: The true labels.
                y_pred: The predicted labels.

            Returns:
                The mean squared error.

                > $MSE = \frac{1}{n} \sum_{i=1}^{n} (y_{true} - y_{pred})^2$
            '''

            return jnp.mean(jnp.power(y_true - y_pred, 2))

        def mse_prime(y_true, y_pred):
            '''
            Implementation of the Mean Squared Error loss function's derivative

            Args:
                y_true: The true labels.
                y_pred: The predicted labels.

            Returns:
                The derivative of the mean squared error.

                > $\frac{\partial MSE}{\partial y_{pred}} = \frac{2}{n} (y_{pred} - y_{true})$
            '''

            return 2 * (y_pred - y_true) / jnp.size(y_true)
        
        super().__init__(mse, mse_prime)



