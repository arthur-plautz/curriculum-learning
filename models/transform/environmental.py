from models.transform.base import BaseTransform
from sklearn.preprocessing import StandardScaler
from pandas import DataFrame

class Environmental(BaseTransform):
    def __init__(self, data:DataFrame, config:dict):
        super().__init__(data, config)

    def reset(self):
        if hasattr(self, 'data') and hasattr(self, 'config'):
            self.__build_performance()
            self.__build_environment()
        if hasattr(self, 'level'):
            self.create_level_label()

    def __build_performance(self):
        col = self.config.get('performance')
        if col in self.data.columns:
            self.performance = self.data[col]
        else:
            raise Exception(f'Column {col} not found for performance in source data')

    def __build_environment(self):
        cols = self.config.get('environment')
        for col in cols:
            if col not in self.data.columns:
                raise Exception(f'Column {col} not found for environment in source data')
        self.environment = self.data[cols]

    def create_level_label(self, level_func=None):
        self.__level_func = level_func if level_func else self.__level_func
        self.level = [self.__level_func(v) for v in self.performance]
        self.data['level'] = self.level
        return self.level

    @property
    def X_normalized(self):
        scaler = StandardScaler().fit(self.environment)
        return scaler.transform(self.environment)

    @property
    def X(self):
        return self.environment

    @property
    def y(self):
        y_column = self.config.get('y_column')
        if y_column == 'level':
            return self.level
        elif y_column == 'performance':
            return self.performance
        else:
            try:
                return self.data[y_column]
            except:
                raise Exception(f'Column {y_column} not found in DataFrame to serve as independent variable')
