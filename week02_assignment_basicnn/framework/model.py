class NeuralNetwork:

    def __init__(self):

        self.layers = []

    def add(self, layer):

        self.layers.append(layer)

    def forward(self, x):

        for layer in self.layers:
            x = layer.forward(x)

        return x

    def backward(self, grad):

        for layer in reversed(self.layers):

            grad = layer.backward(grad)