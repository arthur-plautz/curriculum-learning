import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class DynamicEvolutionStats:
    def __init__(self, seeds, specialist_type):
        self.seeds = seeds
        self.specialist_type = specialist_type
        self.init_data()

    def init_data(self):
        self.data = {}
        for seed in self.seeds:
            self.data[seed] = pd.DataFrame(columns=['generation', 'score', 'cycle'])

    def get_data(self, suffix='score'):
        for seed in self.seeds:
            df = pd.read_csv(f'../../data/specialist/dynamic_evolution/{self.specialist_type}/{seed}_{suffix}.csv')
            self.data[seed] = pd.concat([self.data[seed], df]).query("generation >= 1600")

    def get_seed(self, seed):
        return self.data.get(seed)

    def describe_seeds(self):
        describe = []
        for seed in self.seeds:
            df = self.get_seed(seed)
            describe.append([
                df.score.mean(),
                len(df.query('cycle == "score"')),
                len(df.query('cycle == "fit"')),
            ])

        return pd.DataFrame(
            describe,
            columns=['score', 'score_time', 'fit_time']
        )

    def plot_seeds_scatter(self):
        for seed in self.seeds:
            df = self.get_seed(seed)
            plt.scatter(df.generation, df.score, s=1)
        plt.legend(self.seeds)
        plt.title(f'All Seeds Specialist Score')
        plt.xlabel('generation')
        plt.ylabel('score')
        plt.show()

    def describe_score(self):
        df = self.describe_seeds()
        plt.boxplot(df.score, labels=['mean'])
        plt.title(f'All Seeds Specialist Score Mean')
        plt.ylabel('score')
        plt.show()

    def describe_cycles(self):
        df = self.describe_seeds()
        plt.boxplot(df[['score_time', 'fit_time']], labels=['score_time', 'fit_time'])
        plt.title(f'All Seeds Specialist Cycles')
        plt.xlabel('process')
        plt.ylabel('cycles')
        plt.show()
