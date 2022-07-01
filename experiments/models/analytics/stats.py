import pandas as pd
from models.analytics import NUMERIC_COLUMNS, CM_METRICS, CM_COLUMNS
from models.specialist_data import SpecialistData
from models.utils import *
import warnings
warnings.filterwarnings('ignore')

class StatsAnalytics:
    def __init__(self, data_folder, seeds) -> None:
        self.data_folder = data_folder
        self.target_columns = NUMERIC_COLUMNS
        self.metrics = CM_METRICS
        self.seeds = seeds
        self.load_data()

    def load_data(self):
        self.data = {}
        for seed in self.seeds:
            specialist = SpecialistData(self.data_folder, seed, 'stats', self.target_columns)
            self.data[seed] = specialist

    def get_seed(self, seed):
        return self.data.get(seed)

    def get_summary(self):
        return self.data

    # @property
    # def smaller_length(self):
    #     return min([len(specialist.get_data()) for specialist in self.data.values()])

    def get_mean(self):
        mean_columns = {}
        specialist_data = [specialist.get_data() for specialist in self.data.values()]
        for column in self.target_columns:
            column_group = [specialist[column] for specialist in specialist_data]
            mean_columns[column] = column_group_mean(column_group)
        return pd.DataFrame(mean_columns)

    def get_stats(self, columns):
        if len(columns) == CM_COLUMNS:
            data = self.get_mean()
            return process_cm_metrics(data, columns)
        else:
            raise Exception('Invalid number of Confusion Matrix Columns')
