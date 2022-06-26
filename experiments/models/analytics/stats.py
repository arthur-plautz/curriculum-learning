import pandas as pd
from models.analytics import NUMERIC_COLUMNS, CM_METRICS, CM_COLUMNS
from models.batches_group import BatchesGroup
from models.utils import *
import warnings
warnings.filterwarnings('ignore')

class StatsAnalytics:
    def __init__(self, data_folder, seeds, batch_sizes) -> None:
        self.data_folder = data_folder
        self.target_columns = NUMERIC_COLUMNS
        self.metrics = CM_METRICS
        self.seeds = seeds
        self.batch_sizes = batch_sizes
        self.load_data()

    def load_data(self):
        self.data = {}
        for seed in self.seeds:
            group = BatchesGroup(self.data_folder, self.batch_sizes, seed, 'stats', self.target_columns)
            self.data[seed] = group

    def get_seed(self, seed):
        return self.data.get(seed)

    def get_batch(self, batch):
        data = []
        for seed in self.seeds:
            group = self.get_seed(seed)
            specialist = group.get_batch(batch)
            data.append(specialist.get_data())
        return data

    def get_batch_mean(self, batch):
        mean_columns = {}
        batch_data = self.get_batch(batch)
        for column in self.target_columns:
            column_group = [df[column] for df in batch_data]
            mean_columns[column] = column_group_mean(column_group)
        return pd.DataFrame(mean_columns)

    def get_summary(self):
        summary = {}
        for batch in self.batch_sizes:
            summary[batch] = self.get_batch_mean(batch)
        return summary

    def get_stats(self, columns):
        if len(columns) == CM_COLUMNS:
            data = self.get_summary()

            stats = {}
            for batch in self.batch_sizes:
                stats[batch] = process_cm_metrics(data.get(batch), columns)

            return stats
        else:
            raise Exception('Invalid number of Confusion Matrix Columns')
