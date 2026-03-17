class Trainer:

    def __init__(self, model, loss_fn, optimizer):

        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer

    def train(self, X, y, epochs):

        losses = []

        for epoch in range(epochs):

            y_pred = self.model.forward(X)

            loss = self.loss_fn.forward(y_pred, y)

            grad = self.loss_fn.backward()

            self.model.backward(grad)

            self.optimizer.step(self.model.layers)

            losses.append(loss)

            print(f"Epoch {epoch} Loss {loss:.4f}")

        return losses