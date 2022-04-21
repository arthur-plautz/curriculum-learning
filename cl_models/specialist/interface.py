import time
from inspect import isfunction, ismethod
from sklearn.preprocessing import StandardScaler
from models.specialist.model import SpecialistModel
import pandas as pd

class Specialist:
    def __init__(
        self,
        expected_score=0.7,
        fit_batch_size=20,
        score_batch_size=20,
        start_generation=1
    ):
        self.start_generation = start_generation
        self.generation = start_generation
        self.target_label = None
        self.actual_score = 0
        self.expected_score = expected_score
        self.fit_batch_size = fit_batch_size
        self.score_batch_size = score_batch_size
        self.predict_time = None
        self.model = SpecialistModel()
        self.scaler = StandardScaler()
        self.__fit_counter_reset()

    @property
    def qualified(self):
        # return self.actual_score >= self.expected_score and not self.fit_start
        return False

    @property
    def fit_batch_qualified(self):
        if self.fit_start:
            return (self.generation - self.fit_start) == self.fit_batch_size

    @property
    def score_batch_qualified(self):
        if self.score_start:
            return (self.generation - self.score_start) >= self.score_batch_size

    def __fit_counter_reset(self):
        self.__clear_data()
        self.fit_start = self.generation
        self.score_start = None

    def __score_counter_reset(self):
        self.__clear_data()
        self.fit_start = None
        self.score_start = self.generation

    def __add_data(self, data):
        self.data += data

    def __clear_data(self):
        self.data = []

    def __save_data(self):
        data = []
        X = self.X
        y = self.y
        for i in range(len(X)):
            data.append(X[i] + [y[i]])
        self.__add_data(data)
        return data

    def __transform_data(self, limit=None):
        data = pd.DataFrame(self.data)
        if limit:
            data = data[-(limit-1):]
        x_cols = data.columns[:-1]
        y_col = data.columns[-1]
        X = data[x_cols]
        y = data[y_col]
        return X, y

    def normalize_data(self, data):
        self.scaler = self.scaler.partial_fit(data)
        return self.scaler.transform(data)

    def set_reset_env(self, reset_env):
        if isfunction(reset_env) or ismethod(reset_env):
            self.reset_env = reset_env
        else:
            raise Exception('Reset env must be a function!')

    def set_X(self, X):
        self.X = X

    def set_y(self, y):
        self.y = y

    def set_target_label(self, label):
        self.target_label = label

    def get_labels(self, y):
        return ['good' if p == 1000 else 'bad' for p in y]

    def score(self, X, y):
        labels = self.get_labels(y)
        normalized_X = self.normalize_data(X)
        self.actual_score = self.model.score(normalized_X, labels)

    def fit(self, X, y):
        labels = self.get_labels(y)
        normalized_X = self.normalize_data(X)
        self.model.fit(normalized_X, labels)

    def predict(self):
        if self.target_label:
            start = time.time()
            predicted = None
            seed_salt = 0
            while predicted != self.target_label:
                X = self.reset_env(seed_salt)
                self.set_X([X])
                predicted = self.model.predict(self.X)
                seed_salt += 1
            end = time.time()
            self.predict_time = end - start
            [X] = self.X
            return X
        else:
            raise Exception("Target label is not set!")

    def evaluation(self, X, y):
        self.set_X(X)
        self.set_y(y)

        if self.score_batch_qualified:
            X, y = self.__transform_data(limit=self.score_batch_size)
            self.score(X, y)
            if not self.qualified:
                self.__fit_counter_reset()
        elif self.fit_batch_qualified:
            X, y = self.__transform_data()
            self.fit(X, y)
            self.__score_counter_reset()

        data = self.__save_data()
        self.generation += 1
        return data
