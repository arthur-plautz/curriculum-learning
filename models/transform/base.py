from pandas import DataFrame
from abc import abstractmethod

class BaseTransform:
    def __init__(self, data, config):
        self.set_config(config)
        self.set_data(data)

    def set_config(self, config):
        if isinstance(config, dict):
            self.config = config
            self.reset()
        else:
            raise Exception('Property config is not a dictionary')

    def set_data(self, data):
        if isinstance(data, DataFrame):
            self.data = data
            self.reset()
        else:
            raise Exception('Property data is not a pandas DataFrame')

    @abstractmethod
    def reset(self):
        pass

    @property
    @abstractmethod
    def X(self):
        pass

    @property
    @abstractmethod
    def y(self):
        pass

    @property
    def X_list(self):
        return self.X.values.tolist()

    @property
    def performance_list(self):
        return self.performance.values.tolist()
