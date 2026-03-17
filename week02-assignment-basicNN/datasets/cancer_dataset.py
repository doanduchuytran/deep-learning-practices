from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class CancerDataset:

    def load(self):

        data = load_breast_cancer()

        X = data.data
        y = data.target.reshape(-1,1)

        scaler = StandardScaler()

        X = scaler.fit_transform(X)

        return train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )