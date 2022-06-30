import logging as lg
import pandas as pd
from models.utils import column_group_mean
from models.specialist_data import SpecialistData


class SpecialistsGroup:
    def __init__(self, data_folder, specialists, seed, target_stats, target_columns=[]) -> None:
        self.target_columns = target_columns
        self.target_stats = target_stats
        self.data_folder = data_folder
        self.specialists = specialists
        self.seed = seed
        self.load_data()

    @property
    def greater_specialist_start(self):
        specialists_data = [min(specialist.data.dropna().index) for specialist in self.data.values()] 
        print(max(specialists_data))
        return max(specialists_data)

    def load_data(self):
        self.data = {}
        for specialist_name in self.specialists:
            base_folder = f'{self.data_folder}/{specialist_name}'
            lg.info(f'[Specialist Group] Initializating {specialist_name} ...')
            specialist = SpecialistData(base_folder, self.seed, target_stats=self.target_stats, target_columns=self.target_columns)
            self.data[specialist_name] = specialist
            lg.info(f'[Specialist Group] Finished.')

    def get_specialist(self, specialist_name):
        return self.data.get(specialist_name)

    def get_mean(self, columns):
        mean_columns = {}
        data = [specialist.get_data(start=self.greater_specialist_start) for specialist in self.data.values()]
        for column in columns:
            column_group = [df[column] for df in data]
            mean_columns[column] = column_group_mean(column_group)
        return pd.DataFrame(mean_columns)
