import pandas as pd
from models.specialist_data import SpecialistData
from models.utils import column_group_mean
import warnings
warnings.filterwarnings('ignore')

class BaseConditionsAnalytics:
    def __init__(self, data_folder, seeds) -> None:
        self.data_folder = data_folder
        self.target_columns = [str(i) for i in range(729)]
        self.seeds = seeds
        self.load_data()

    def load_data(self):
        self.data = {}
        for seed in self.seeds:
            specialist = SpecialistData(self.data_folder, seed, 'base_conditions', self.target_columns)
            self.data[seed] = specialist

    def get_summary(self):
        return self.data

    def get_seed(self, seed):
        return self.data.get(seed)

    def get_mean(self):
        mean_columns = {}
        specialist_data = [specialist.get_data() for specialist in self.data.values()]
        for column in self.target_columns:
            column_group = [specialist[column] for specialist in specialist_data]
            mean_columns[column] = column_group_mean(column_group)
        return pd.DataFrame(mean_columns)
