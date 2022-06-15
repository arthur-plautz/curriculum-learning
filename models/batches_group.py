import pandas as pd
from models.utils import column_group_mean
from models.specialist_data import SpecialistData


class BatchesGroup:
    def __init__(self, data_folder, batch_sizes, seed) -> None:
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
            base_folder = f'{self.data_folder}/specialist_sp_g{batch}'
            specialist = SpecialistData(base_folder, self.seed)
            self.data[batch] = specialist

    def get_mean(self, columns):
        mean_columns = {}
        data = [specialist.get_data(start=self.greater_batch_size) for specialist in self.data.values()]
        for column in columns:
            column_group = [df[column] for df in data]
            mean_columns[column] = column_group_mean(column_group)
        return pd.DataFrame(mean_columns)
