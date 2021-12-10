import pandas as pd
import matplotlib.pyplot as plt

class ModelStats:
    def __init__(self, seeds, batch_sizes):
        self.seeds = seeds
        self.batch_sizes = batch_sizes
        self.init_data()

    def init_data(self):
        self.data = {}
        for seed in self.seeds:
            self.data[seed] = pd.DataFrame(columns=['batch', 'stage', 'score'])

    def get_data(self):
        for batch_size in self.batch_sizes:
            for seed in self.seeds:
                df = pd.read_csv(f'../../data/specialist/evolution/{batch_size}_batch_{seed}_score.csv')
                df['batch'] = batch_size
                self.data[seed] = pd.concat([self.data[seed], df])

    def get_seed(self, seed):
        return self.data.get(seed)

    def get_batch(self, batch):
        result = pd.DataFrame(columns=['seed', 'stage', 'score'])
        for seed in self.seeds:
            df = self.get_seed(seed)
            df = df.query(f'batch == {batch}')
            df['seed'] = seed
            result = pd.concat([result, df])
        return result

    def describe_batch(self, batch):
        describe = []
        batch_df = self.get_batch(batch)
        for seed in self.seeds:
            df = batch_df.query(f'seed == "{seed}"')
            describe.append([
                seed,
                df.score.mean(),
                df.score.skew(),
                df.score.std(),
                df.score.min(),
                df.score.max()
            ])

        return pd.DataFrame(
            describe,
            columns=['seed', 'mean', 'skew', 'std', 'min', 'max']
        )

    def plot_seeds_scatter(self, batch):
        batch_df = self.get_batch(batch)
        for seed in self.seeds:
            df = batch_df.query(f'seed == "{seed}"')
            plt.scatter(df.stage, df.score, s=1)
        plt.legend(self.seeds)
        plt.show()

    def compare_batches_metric(self, metric):
        results = []
        for batch in self.batch_sizes:
            df = self.describe_batch(batch)
            results.append(df[metric])
        plt.boxplot(results, labels=self.batch_sizes)
        plt.show()
