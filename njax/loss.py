class Loss:
    '''
    Parent class for all loss functions.
    It is used to store the loss function and its derivative.
    It is also used to call the loss function and its derivative when needed.
    '''

    def __init__(self, loss, loss_prime):
        """
        Args:
            loss: The loss function.
            loss_prime: The loss function's derivative.
        """
        # Store the loss function and its derivative
        self.loss = loss
        self.loss_prime = loss_prime

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
