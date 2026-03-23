from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class HousingDataset:

    def load(self):

        data = fetch_california_housing()

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