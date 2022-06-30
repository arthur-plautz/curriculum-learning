import pandas as pd
from models.analytics import FIT_EVALUATION_COLUMNS, CM_METRICS, CM_COLUMNS
from models.specialists_group import SpecialistsGroup
from models.utils import *
import warnings
warnings.filterwarnings('ignore')

class FitEvaluationGroupAnalytics:
    def __init__(self, data_folder, seeds, specialists) -> None:
        self.data_folder = data_folder
        self.target_columns = FIT_EVALUATION_COLUMNS
        self.metrics = CM_METRICS
        self.seeds = seeds
        self.specialists = specialists
        self.load_data()

    def load_data(self):
        self.data = {}
        for seed in self.seeds:
            group = SpecialistsGroup(self.data_folder, self.specialists, seed, 'fit_evaluation')
            self.data[seed] = group

    def get_seed(self, seed):
        return self.data.get(seed)

    def get_specialist(self, specialist_name):
        data = []
        for seed in self.seeds:
            group = self.get_seed(seed)
            specialist = group.get_specialist(specialist_name)
            data.append(specialist.get_data())
        return data

    def get_specialist_mean(self, specialist):
        mean_columns = {}
        specialist_data = self.get_specialist(specialist)
        for column in self.target_columns:
            column_group = [df[column] for df in specialist_data]
            mean_columns[column] = column_group_mean(column_group)
        return pd.DataFrame(mean_columns)

    def get_summary(self):
        summary = {}
        for specialist in self.specialists:
            summary[specialist] = self.get_specialist_mean(specialist)
        return summary

    def get_stats(self, columns):
        if len(columns) == CM_COLUMNS:
            data = self.get_summary()

            stats = {}
            for batch in self.specialists:
                stats[batch] = process_cm_metrics(data.get(batch), columns)

            return stats
        else:
            raise Exception('Invalid number of Confusion Matrix Columns')
