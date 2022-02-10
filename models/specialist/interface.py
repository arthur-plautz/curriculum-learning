import time
from inspect import isfunction, ismethod
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import pandas as pd

class Specialist:
    def __init__(
        self,
        expected_score=0.85,
        labels=['bad', 'good'],
        batch_size=20,
        n_trials=10,
        start_generation=1
    ):
        self.process_counter = start_generation
        self.labels = labels
        self.target_label = None
        self.actual_score = 0
        self.expected_score = expected_score
        self.n_trials = n_trials
        self.batch_size = batch_size
        self.predict_time = None
        self.set_classifier()
        self.set_scaler()
        self.__fit_counter_reset()

    @property
    def generation(self):
        return self.process_counter // self.n_trials

    @property
    def qualified(self):
        return self.actual_score >= self.expected_score

    @property
    def fit_batch_qualified(self):
        if self.fit_start:
            return (self.process_counter - self.fit_start) == self.batch_size * self.n_trials

    @property
    def score_batch_qualified(self):
        if self.score_start:
            return (self.process_counter - self.score_start) >= self.batch_size * self.n_trials

    @property
    def params(self):
        return {
            'activation': 'tanh',
            'alpha': 0.0001,
            'hidden_layer_sizes': (64, 32, 64, 32),
            'solver': 'sgd'
        }

    def __fit_counter_reset(self):
        self.__clear_data()
        self.fit_start = self.process_counter
        self.score_start = None

    def __score_counter_reset(self):
        self.__clear_data()
        self.fit_start = None
        self.score_start = self.process_counter

    def __add_data(self, data):
        self.data.append(data)

    def __clear_data(self):
        self.data = []

    def __save_data(self):
        [X] = self.X.values.tolist()
        y = self.y
        data = X + y
        self.__add_data(data)

    def __transform_data(self):
        data = pd.DataFrame(self.data)
        x_cols = data.columns[:-1]
        y_col = data.columns[-1]
        X = data[x_cols]
        y = data[y_col]
        return X, y

    def __verify_data(self, data):
        if not isinstance(data, pd.DataFrame):
            raise Exception('Data provided must be a pandas dataframe!')
        elif data.empty:
            raise Exception('Data provided is empty!')

    def normalize_data(self, data):
        self.scaler = self.scaler.partial_fit(data)
        return self.scaler.transform(data)

    def set_classifier(self):
        self.clf = MLPClassifier(
            random_state=42,
            **self.params
        )
    
    def set_scaler(self):
        self.scaler = StandardScaler()

    def set_reset_env(self, reset_env):
        if isfunction(reset_env) or ismethod(reset_env):
            self.reset_env = reset_env
        else:
            raise Exception('Reset env must be a function!')

    def set_X(self, X):
        self.__verify_data(X)
        self.X = X

    def set_y(self, y):
        self.y = y

    def set_target_label(self, label):
        self.target_label = label

    def score(self, X, y):
        X_batch = self.normalize_data(X[-self.batch_size:])
        y_batch = y[-self.batch_size:]
        self.actual_score = self.clf.score(X_batch, y_batch)

    def fit(self, X, y):
        X = self.normalize_data(X)
        self.clf = self.clf.partial_fit(X, y, self.labels)

    def predict(self):
        if self.target_label:
            start = time.time()
            predicted = None
            while predicted != self.target_label:
                X = self.reset_env()
                self.set_X(X)
                norm_X = self.normalize_data(X)
                predicted = self.clf.predict(norm_X)
            end = time.time()
            self.predict_time = end - start
            return predicted
        else:
            raise Exception("Target label is not set!")

    def pre_process(self, X):
        self.set_X(X)

        if self.qualified and not self.fit_start:
            return self.predict()

    def post_process(self, y):
        self.set_y(y)

        if self.score_batch_qualified:
            X, y = self.__transform_data()
            self.score(X, y)
            if not self.qualified:
                self.__fit_counter_reset()
        elif self.fit_batch_qualified:
            X, y = self.__transform_data()
            self.fit(X, y)
            self.__score_counter_reset()

        self.__save_data()
        self.process_counter += 1