# Import jit from JAX
from jax import jit
from NeuralJAXwork import Errors

class Loss:
    """
    Parent class for all loss functions.
    It is used to store the loss function and its derivative.
    It is also used to call the loss function and its derivative when needed.
    """

    def __init__(self, loss, loss_prime):
        """ Initialize the loss function and its derivative.

        It also compiles the loss function and its derivative with JAX's jit
        to speed up the training process
        and store it

        Args:
            loss: The loss function.
            loss_prime: The loss function's derivative.
        
        Raises:
            Exception: If the loss function or its derivative could not be compiled with JAX's jit.
        """

        # Try to compile the loss function
        # If it fails, switch to regular Python interpreator
        try:
            self.loss = jit(loss)
        except:
            print(Errors.jit_error)
            self.loss = loss
        

        # Try to compile the loss function's derivative
        # If it fails, switch to regular Python interpreator
        try:
            self.loss_prime = jit(loss_prime)
        except:
            print(Errors.jit_error)
            self.loss = loss_prime

    def loss(self, y_true, y_pred):
        """
        Call the loss function.

        Args:
            y_true: The true labels.
            y_pred: The predicted labels.

        Returns:
            The loss function's output.
        """
        # Return the loss function
        return self.loss(y_true, y_pred)


    def prime(self, y_true, y_pred):
        """
        Call the loss function's derivative.

        Args:
            y_true: The true labels.
            y_pred: The predicted labels.
        
        Returns:
            The loss function's derivative's output.
        """
        # Return the loss function's derivative
        return self.loss_prime(y_true, y_pred)
