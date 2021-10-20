from pandas import DataFrame
from abc import abstractmethod

class BaseTransform:
    def __init__(self, data, config):
        self.__set_data(data)
        self.__set_config(config)

    def __set_config(self, config):
        if isinstance(config, dict):
            self.config = config
        else:
            raise Exception('Property config is not a dictionary')

    def __set_data(self, data):
        if isinstance(data, DataFrame):
            self.data = data
        else:
            raise Exception('Property data is not a pandas DataFrame')
    @property
    @abstractmethod
    def X(self):
        pass

    @property
    @abstractmethod
    def y(self):
        pass
