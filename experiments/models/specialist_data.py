import os
import pathlib
import pandas as pd
import logging as lg

MODEL_DIR = str(pathlib.Path(__file__).parent.resolve())

class SpecialistData:
    def __init__(self, data_folder, seed, target_stats, target_columns=[]):
        self.target_stats = target_stats
        self.target_columns = target_columns
        self.seed = seed
        self.data_path = f'{self.__data_root}/{data_folder}_{self.target_stats}/{self.seed}_run.csv'
        self.read_data()

    @property
    def data_path(self):
        return self.__data_path

    @data_path.setter
    def data_path(self, data_path):
        if os.path.isfile(data_path):
            self.__data_path = data_path
        else:
            raise Exception(f'Directory [{data_path}] not found.')

    @property
    def __data_root(self):
        return MODEL_DIR.replace('experiments/models', 'data')

    def read_data(self):
        lg.info(f'[Specialist Manager] Reading Specialist [{self.seed}] Data ...')
        data = pd.read_csv(self.data_path, index_col=False)
        lg.info('[Specialist Manager] Finished.')
        if self.target_columns:
            data = data[self.target_columns]
        self.data = data

    def get_data(self, start=0, end=None, drop_mode='any'):
        data = self.data[start:].dropna(how=drop_mode)
        if end:
            return data[:end]
        else:
            return data
