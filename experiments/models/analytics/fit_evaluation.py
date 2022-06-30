import pandas as pd
from models.analytics import NUMERIC_COLUMNS, CM_METRICS, CM_COLUMNS
from models.specialist_data import SpecialistData
from models.utils import *
import warnings
warnings.filterwarnings('ignore')

class FitEvaluationAnalytics:
    def __init__(self, data_folder, seeds) -> None:
        self.data_folder = data_folder
        self.target_columns = NUMERIC_COLUMNS
        self.metrics = CM_METRICS
        self.seeds = seeds
        self.load_data()

    def load_data(self):
        self.data = {}
        for seed in self.seeds:
            specialist = SpecialistData(self.data_folder, seed, 'fit_evaluation', self.target_columns)
            self.data[seed] = specialist

    def get_seed(self, seed):
        return self.data.get(seed)

    def get_summary(self):
        return self.data

    def get_mean(self):
        mean_columns = {}
        for column in self.target_columns:
            column_group = [df[column] for df in self.data.values()]
            mean_columns[column] = column_group_mean(column_group)
        return pd.DataFrame(mean_columns)

    def get_stats(self, columns):
        if len(columns) == CM_COLUMNS:
            data = self.get_mean()
            return process_cm_metrics(data, columns)
        else:
            raise Exception('Invalid number of Confusion Matrix Columns')
