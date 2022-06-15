from tracemalloc import start
import numpy as np
from models.specialist_data import SpecialistData


class ExperimentGroup:
    def __init__(self, data_folder, name, batch_sizes, seeds) -> None:
        self.data_folder = data_folder
        self.name = name
        self.batch_sizes = batch_sizes
        self.seeds = seeds
        self.get_group()

    def get_group(self):
        self.group = {}
        for batch in self.batch_sizes:
            self.group.update(batch, {})
            base_folder = f'{self.data_folder}/specialist_sp_g{batch}'
            for seed in self.seeds:
                specialist = SpecialistData(base_folder, seed)
                self.group[batch][seed] = specialist

    def get_column_mean(self, column_arr):
        if len(column_arr) > 1:
            m = np.mean(column_arr)
            return m[~np.isnan(m)]
        else:
            return column_arr[0]
