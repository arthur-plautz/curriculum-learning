import pandas as pd
from models.agent_data import AgentData
from models.utils import column_group_mean
import warnings
warnings.filterwarnings('ignore')

class AgentBaseConditionsAnalytics:
    def __init__(self, data_folder, seeds) -> None:
        self.data_folder = data_folder
        self.target_columns = [str(i) for i in range(729)]
        self.seeds = seeds
        self.load_data()

    def load_data(self):
        self.data = {}
        for seed in self.seeds:
            agent = AgentData(self.data_folder, seed, self.target_columns)
            self.data[seed] = agent

    def get_seed(self, seed):
        return self.data.get(seed)

    @property
    def smaller_length(self):
        return min([len(self.get_seed(seed).data) for seed in self.seeds])

    def get_seed(self, seed):
        return self.data.get(seed)

    def get_summary(self):
        summary = []
        for seed in self.seeds:
            agent = self.get_seed(seed)
            summary.append(
                agent.get_data(end=self.smaller_length)
            )
        return summary

    def get_mean(self):
        mean_columns = {}
        data = self.get_summary()
        for column in self.target_columns:
            column_group = [df[column] for df in data]
            mean_columns[column] = column_group_mean(column_group)
        return pd.DataFrame(mean_columns)
