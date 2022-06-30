import pandas as pd
from models.specialists_group import SpecialistsGroup
from models.utils import column_group_mean
import warnings
warnings.filterwarnings('ignore')

class BaseConditionsGroupAnalytics:
    def __init__(self, data_folder, seeds, batch_sizes) -> None:
        self.data_folder = data_folder
        self.target_columns = [str(i) for i in range(729)]
        self.seeds = seeds
        self.batch_sizes = batch_sizes
        self.load_data()

    def load_data(self):
        self.data = {}
        for seed in self.seeds:
            group = SpecialistsGroup(self.data_folder, self.batch_sizes, seed, 'base_conditions', self.target_columns)
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

    @property
    def smaller_length(self):
        return min([len(self.get_seed(seed).data) for seed in self.seeds])

    def get_seed(self, seed):
        return self.data.get(seed)

    def get_summary(self):
        summary = []
        for seed in self.seeds:
            data = self.get_seed(seed)
            summary.append(
                data.get_mean(self.target_columns)
            )
        return summary

    def get_mean(self):
        mean_columns = {}
        data = self.get_summary()
        for column in self.target_columns:
            column_group = [df[column] for df in data]
            mean_columns[column] = column_group_mean(column_group)
        return pd.DataFrame(mean_columns)
