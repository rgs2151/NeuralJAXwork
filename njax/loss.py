# Import jit from JAX
from jax import jit

class Loss:
    """
    Parent class for all loss functions.
    It is used to store the loss function and its derivative.
    It is also used to call the loss function and its derivative when needed.
    """

    def __init__(self, loss, loss_prime):
        """
        Args:
            loss: The loss function.
            loss_prime: The loss function's derivative.
        """
        
        # Compile the loss function with JAX's jit
        # to speed up the training process
        # and store it

        # Try to compile the loss function
        # If it fails, raise an error
        try:
            self.loss = jit(loss)
        except:
            raise Exception("NJAX: The loss function could not be compiled with JAX's jit.")

        # Compile the loss function's derivative with JAX's jit
        # to speed up the training process
        # and store it

        # Try to compile the loss function's derivative
        # If it fails, raise an error
        try:
            self.loss_prime = jit(loss_prime)
        except:
            raise Exception("NJAX: The loss function's derivative could not be compiled with JAX's jit.")

    def __call__(self, y_true, y_pred):
        """
        Call the loss function.

        Args:
            y_true: The true labels.
            y_pred: The predicted labels.
        """
        # Return the loss function
        return self.loss(y_true, y_pred)


    def prime(self, y_true, y_pred):
        """
        Call the loss function's derivative.

        Args:
            y_true: The true labels.
            y_pred: The predicted labels.
        """
        # Return the loss function's derivative
        return self.loss_prime(y_true, y_pred)
