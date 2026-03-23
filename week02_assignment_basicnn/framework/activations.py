import numpy as np


class ReLU:

    def forward(self, x):

        self.x = x
        return np.maximum(0, x)

    def backward(self, grad):

        grad[self.x <= 0] = 0
        return grad


class Sigmoid:

    def forward(self, x):

        self.out = 1 / (1 + np.exp(-x))
        return self.out

    def backward(self, grad):

        return grad * self.out * (1 - self.out)


class Softmax:

    def forward(self, x):

        exp = np.exp(x - np.max(x, axis=1, keepdims=True))
        self.out = exp / np.sum(exp, axis=1, keepdims=True)

        return self.out

    def backward(self, grad):

        return grad