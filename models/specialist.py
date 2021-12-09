from sklearn.neural_network import MLPClassifier
import warnings
import pandas as pd
warnings.filterwarnings('ignore')

class Specialist:
    def __init__(self, transformed, seed):
        self.seed = seed
        self.transformed = transformed
        self.data = transformed.data.copy()
        self.set_classifier()

    def set_classifier(self):
        self.clf = MLPClassifier(
            hidden_layer_sizes=(64,64,64,64,64),
            alpha=0.000001,
            max_iter=2000,
            activation="tanh",
            verbose=10,
            random_state=42,
            tol=0.000005
        )

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
        self.clf = self.clf.partial_fit(self.transformed.X, self.transformed.level, ['bad', 'good'])

    def test_model(self):
        test_data = self.get_batch(self.batch_middle, self.batch_end)
        self.transformed.set_data(test_data)
        return self.clf.score(self.transformed.X, self.transformed.level)

    def evolve_stage(self):
        self.train_model()
        return self.test_model()

    def evolve_process(self, interval=1, min_limit=15, max_limit=40):
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
        df.to_csv(f'../../data/specialist/evolution/{int(self.batch_size)}_batch_{self.seed}_score.csv', index=False)
