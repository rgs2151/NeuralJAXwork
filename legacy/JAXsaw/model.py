class Model:

    def configure(self, layers = [], loss = None):
        self.layers = layers
        self.loss = loss
        pass
    
    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad
    
    def train(self, x_train, y_train, epochs = 1000, learning_rate = 0.01, verbose = True):
        for e in range(epochs):
            error = 0
            for x, y in zip(x_train, y_train):
                # forward
                output = self.forward(x)

                # error
                error += self.loss.loss(y, output)

                # backward
                grad = self.loss.prime(y, output)
                self.backward(grad)

            error /= len(x_train)

            if verbose:
                print(f"{e + 1}/{epochs}, error={error}")

    def __call__(self, x):
        return self.forward(x)

    def __repr__(self):
        return f'SequentialModel({zip(enumerate(self.layers))})'