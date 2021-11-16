from models.transform.base import BaseTransform
from pandas import DataFrame

class Environmental(BaseTransform):
    def __init__(self, data:DataFrame, config:dict):
        super().__init__(data, config)
        self.__build_performance()
        self.__build_environment()

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

    def __level_label(self, value, min_value=0, max_value=100):
        interval = max_value - min_value
        value -= min_value
        if value <= interval*0.25:
            return 'bad'
        elif value <= interval*0.5:
            return 'medium'
        elif value <= interval*0.75:
            return 'good'
        else:
            return 'top'

    def create_level_label(self):
        min_v = self.data.performance.min()
        max_v = self.data.performance.max()
        self.level = [self.__level_label(v, min_v, max_v) for v in self.performance]
        self.data['level'] = self.level
        return self.level

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
