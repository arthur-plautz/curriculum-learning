import logging as lg
import pandas as pd
from models.utils import column_group_mean
from models.specialist_data import SpecialistData


class BatchesGroup:
    def __init__(self, data_folder, batch_sizes, seed, target_stats, target_columns=[]) -> None:
        self.target_columns = target_columns
        self.target_stats = target_stats
        self.data_folder = data_folder
        self.batch_sizes = batch_sizes
        self.seed = seed
        self.load_data()

    @property
    def greater_batch_size(self):
        return max(self.batch_sizes)

    def load_data(self):
        self.data = {}
        for batch in self.batch_sizes:
            base_folder = f'{self.data_folder}/sp{batch}'
            lg.info(f'[Batch Manager] Initializating Batch {batch} ...')
            specialist = SpecialistData(base_folder, self.seed, target_stats=self.target_stats, target_columns=self.target_columns)
            self.data[batch] = specialist
            lg.info(f'[Batch Manager] Finished.')

    def get_batch(self, batch):
        return self.data.get(batch)

    def get_mean(self, columns):
        mean_columns = {}
        data = [specialist.get_data(start=self.greater_batch_size) for specialist in self.data.values()]
        for column in columns:
            column_group = [df[column] for df in data]
            mean_columns[column] = column_group_mean(column_group)
        return pd.DataFrame(mean_columns)
