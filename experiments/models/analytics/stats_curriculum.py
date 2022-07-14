import pandas as pd
from models.curriculum_data import CurriculumData
from models.utils import *
import warnings
warnings.filterwarnings('ignore')

class CurriculumStats:
    def __init__(self, curriculum_folder, seeds) -> None:
        self.data_folder = f'{curriculum_folder}/runstats'
        self.target_columns = ['msteps','bestfit','bestgfit','bestsam','avgfit','paramsize','gen']
        self.seeds = seeds
        self.load_data()

    def load_data(self):
        self.data = {}
        for seed in self.seeds:
            curriculum = CurriculumData(self.data_folder, seed)
            self.data[seed] = curriculum

    def get_seed(self, seed):
        return self.data.get(seed)

    def get_summary(self):
        return self.data

    def get_mean(self):
        mean_columns = {}
        specialist_data = [specialist.get_data() for specialist in self.data.values()]
        for column in self.target_columns:
            column_group = [specialist[column] for specialist in specialist_data]
            mean_columns[column] = column_group_mean(column_group)
        return pd.DataFrame(mean_columns)
