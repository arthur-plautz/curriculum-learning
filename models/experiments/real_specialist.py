import pandas as pd
from models.experiments import NUMERIC_COLUMNS
from models.batches_group import BatchesGroup
from models.utils import column_group_mean
import warnings
warnings.filterwarnings('ignore')

class RealSpecialistExperiment:
    def __init__(self, data_folder, seeds, batch_sizes) -> None:
        self.data_folder = data_folder
        self.target_columns = NUMERIC_COLUMNS
        self.seeds = seeds
        self.batch_sizes = batch_sizes
        self.load_data()

    def load_data(self):
        self.data = {}
        for seed in self.seeds:
            group = BatchesGroup(self.data_folder, self.batch_sizes, seed)
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
