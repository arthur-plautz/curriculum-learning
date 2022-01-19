import time
from inspect import isfunction, ismethod
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import pandas as pd

class Specialist:
    def __init__(
        self,
        expected_score=0.8,
        labels=['bad', 'good'],
        batch_size=50
    ):
        self.labels = labels
        self.target_label = None
        self.real_score = 0
        self.expected_score = expected_score
        self.open_batch = 0
        self.batch_size = batch_size
        self.predict_time = None
        self.set_classifier()

    @property
    def qualified(self):
        return self.real_score >= self.expected_score

    @property
    def batch_qualified(self):
        return self.open_batch == self.batch_size

    @property
    def params(self):
        return {
            'random_state': 42,
            'activation': 'tanh',
            'alpha': 0.0001,
            'hidden_layer_sizes': (64, 32, 64, 32),
            'solver': 'sgd'
        }

    def __add_data(self, data):
        self.data.append(data)

    def __clear_data(self):
        self.data = []

    def __verify_data(self, data):
        if not isinstance(data, pd.DataFrame):
            raise Exception('Data provided must be a pandas dataframe!')
        elif data.empty:
            raise Exception('Data provided is empty!')
        elif data.isna():
            raise Exception('Data provided contain NaN values!')

    def __process_data(self, data, transform=False):
        self.__verify_data(data)
        if transform:
            scaler = StandardScaler().fit(data)
            return scaler.transform(data)

    def set_classifier(self):
        self.clf = MLPClassifier(
            **self.params
        )

    def set_reset_env(self, reset_env):
        if isfunction(reset_env) or ismethod(reset_env):
            self.reset_env = reset_env
        else:
            raise Exception('Reset env must be a function!')

    def set_X(self, X):
        self.X = self.__process_data(X, transform=True)

    def set_y(self, y):
        self.y = self.__process_data(y)

    def set_target_label(self, label):
        self.target_label = label

    def set_score(self, X=None, y=None):
        X = X if X else self.X
        y = y if y else self.y
        self.real_score = self.clf.score(X, y)

    def fit(self, X=None, y=None):
        X = X if X else self.X
        y = y if y else self.y
        self.clf.partial_fit(X, y, self.labels)

    def predict(self, X=None):
        X = X if X else self.X
        if self.target_label:
            start = time.time()
            predicted = None
            while predicted != self.target_label:
                X = self.self.reset_env()
                self.set_X(X)
                predicted = self.clf.predict(self.X)
            end = time.time()
            self.predict_time = end - start
            return predicted
        else:
            raise Exception("Target label is not set!")

    def pre_process(self, X):
        self.set_X(X)

        if not bool(self.open_batch):
            if self.qualified:
                return self.predict()
            else:
                self.open_batch = 1
        else:
            self.open_batch += 1

    def post_process(self, y):
        self.set_y(y)

        if not bool(self.open_batch):
            self.score()
        else:
            if self.batch_qualified:
                data = pd.DataFrame(self.data)
                x_cols = data.columns[:-1]
                y_col = data.columns[-1]
                self.fit(
                    X=data[x_cols],
                    y=data[y_col]
                )
                self.__clear_data()
            else:
                X = self.X.copy()
                y = self.y.copy()
                data = X + [y]
                self.__add_data(data)
