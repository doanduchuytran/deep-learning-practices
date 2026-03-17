import numpy as np


class MSELoss:

    def forward(self, y_pred, y_true):

        self.y_pred = y_pred
        self.y_true = y_true

        return np.mean((y_pred - y_true) ** 2)

    def backward(self):

        n = self.y_true.shape[0]

        return 2 * (self.y_pred - self.y_true) / n


class BinaryCrossEntropy:

    def forward(self, y_pred, y_true):

        self.y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
        self.y_true = y_true

        loss = -np.mean(
            y_true * np.log(self.y_pred)
            + (1 - y_true) * np.log(1 - self.y_pred)
        )

        return loss

    def backward(self):

        n = self.y_true.shape[0]

        return (self.y_pred - self.y_true) / n


class CrossEntropy:

    def forward(self, y_pred, y_true):

        self.y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
        self.y_true = y_true

        n = y_true.shape[0]

        return -np.sum(y_true * np.log(self.y_pred)) / n

    def backward(self):

        n = self.y_true.shape[0]

        return (self.y_pred - self.y_true) / n