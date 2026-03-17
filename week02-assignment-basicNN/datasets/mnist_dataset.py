from sklearn.datasets import fetch_openml
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import numpy as np

class MNISTDataset:
    def load(self):
        # Load MNIST as NumPy arrays
        X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
        X = X / 255.0

        # One-hot encode labels
        encoder = OneHotEncoder(sparse=False)  # works for scikit-learn <1.2
        y = encoder.fit_transform(y.reshape(-1,1))

        # Split dataset
        return train_test_split(
            np.array(X, dtype=np.float32),
            np.array(y, dtype=np.float32),
            test_size=0.2,
            random_state=42
        )