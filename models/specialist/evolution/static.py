import os
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

class StaticEvolution:
    def __init__(self, transformed, seed, specialist):
        self.seed = seed
        self.transformed = transformed
        self.data = transformed.data.copy()
        self.specialist = specialist
        self.save_path = f'../../data/specialist/static_evolution/{self.specialist.type}'

    def verify_dir(self):
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)

    @property
    def batch_start(self):
        return self.batch_stg * self.batch_size

    @property
    def batch_middle(self):
        return self.batch_start + self.batch_size

    @property
    def batch_end(self):
        return self.batch_middle + self.batch_size

    def get_batch(self, lower_limit, upper_limit):
        return self.data.query(f'index > {lower_limit} and index < {upper_limit}')

    def train_model(self):
        train_data = self.get_batch(self.batch_start, self.batch_middle)
        self.transformed.set_data(train_data)
        self.specialist.fit(self.transformed.X_normalized, self.transformed.y)

    def test_model(self):
        test_data = self.get_batch(self.batch_middle, self.batch_end)
        self.transformed.set_data(test_data)
        self.specialist.score(self.transformed.X_normalized, self.transformed.y)
        return self.specialist.actual_score

    def evolve_stage(self):
        self.train_model()
        return self.test_model()

    def evolve_process(self, interval=1, min_limit=10, max_limit=40, suffix='score'):
        results = []
        stages = []
        to_int = 1/interval
        max_gen = max_limit - (interval * to_int)
        min_gen = min_limit * to_int
        self.batch_size = 1000 * interval
        for i in range(int(min_gen), int(max_gen*to_int), int(interval*to_int)):
            self.batch_stg = i
            result = self.evolve_stage()
            results.append(int(result*100))
            stages.append(self.batch_start/10)
        df = pd.DataFrame({'stage': stages, 'score': results})
        df.to_csv(f'{self.save_path}/{int(self.batch_size)}_bs_{self.seed}_{suffix}.csv', index=False)
