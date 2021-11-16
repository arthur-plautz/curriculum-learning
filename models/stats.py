import pandas as pd
import matplotlib.pyplot as plt

class Stats:
    def __init__(self, data_dir, seed):
        self.data_dir = data_dir
        self.stats_dir = '/runstats'
        self.seed = seed
        self.get_metrics_data()
        self.get_run_data()

    def get_metrics_data(self):
        metrics_file = f'{self.data_dir}{self.stats_dir}/s{self.seed}_metrics.csv'
        self.metrics_data = pd.read_csv(metrics_file, index_col=False)
        self.metrics_data = self.metrics_data.dropna()

    def get_run_data(self):
        run_file = f'{self.data_dir}{self.stats_dir}/s{self.seed}_run.csv'
        self.run_data = pd.read_csv(run_file, index_col=False)
        self.run_data = self.run_data.dropna()

    def fitness_evolution(self):
        plt.plot(self.run_data.index, self.run_data.avgfit)
        plt.show()
