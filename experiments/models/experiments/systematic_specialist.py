import pandas as pd
from models.experiments import BASE_COLUMNS
from models.specialist_data import SpecialistData
from models.utils import column_group_mean
import warnings
warnings.filterwarnings('ignore')

class SystematicSpecialistExperiment:
    def __init__(self, data_folder, seeds) -> None:
        self.data_folder = data_folder
        self.target_columns = BASE_COLUMNS
        self.seeds = seeds
        self.load_data()

    def load_data(self):
        self.data = {}
        for seed in self.seeds:
            base_path = f'{self.data_folder}/specialist_sp_base'
            specialist_data = SpecialistData(base_path, seed, self.target_columns)
            self.data[seed] = specialist_data

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
                data.get_data(end=self.smaller_length)
            )
        return summary

    def get_mean(self):
        mean_columns = {}
        data = self.get_summary()
        for column in self.target_columns:
            column_group = [df[column] for df in data]
            mean_columns[column] = column_group_mean(column_group)
        return pd.DataFrame(mean_columns)
