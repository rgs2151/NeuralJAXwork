from NeuralJAXwork import Model

class SequentialModel(Model):
    
    def __init__(self, layers = [], loss = None):

        self.layers = layers
        self.loss = loss

        super().configure(self.layers, self.loss)

    def add(self, layer):
        super().layers.append(layer)
    
    def remove(self, index):
        super().layers.remove(index)
