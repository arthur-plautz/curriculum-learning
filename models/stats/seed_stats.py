import pandas as pd
import matplotlib.pyplot as plt

class SeedStats:
    def __init__(self, data_dir, seed):
        self.seed = seed
        self.data_dir = data_dir
        self.stats_dir = '/runstats'
        self.get_metrics_data()
        self.get_run_data()

    def clear_data(self, data, positive_cols=[]):
        data = data.dropna()
        for col in positive_cols:
            data[col] = pd.to_numeric(data[col], errors='coerce')
            data = data.query(f"{col} >= 0")
        return data

    def get_metrics_data(self):
        metrics_file = f'{self.data_dir}{self.stats_dir}/s{self.seed}_metrics.csv'
        self.metrics_data = pd.read_csv(metrics_file, index_col=False)
        self.metrics_data = self.clear_data(
            self.metrics_data
        )

    def get_test_data(self):
        test_file = f'{self.data_dir}{self.stats_dir}/s{self.seed}_test.csv'
        self.test_data = pd.read_csv(test_file, index_col=False)
        self.test_data = self.test_data.dropna()

    def get_run_data(self):
        run_file = f'{self.data_dir}{self.stats_dir}/s{self.seed}_run.csv'
        verify_cols = ['bestfit', 'bestgfit', 'avgfit']
        self.run_data = pd.read_csv(run_file, index_col=False)
        self.run_data = self.clear_data(
            self.run_data,
            verify_cols
        )

    def fitness_evolution(self, show=True, period=1):
        data = self.run_data.groupby(self.run_data.index // period).mean()
        plt.plot(data.index, data.avgfit)
        if show:
            plt.show()
