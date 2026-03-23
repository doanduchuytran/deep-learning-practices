import numpy as np

class Linear:

    def __init__(self, input_dim, output_dim):

        self.W = np.random.randn(input_dim, output_dim) * 0.01
        self.b = np.zeros((1, output_dim))

    def forward(self, x):

        self.x = x
        return x @ self.W + self.b

    def backward(self, grad):

        self.dW = self.x.T @ grad
        self.db = np.sum(grad, axis=0, keepdims=True)

        dx = grad @ self.W.T

        return dx