import pandas as pd
import warnings
warnings.filterwarnings('ignore')

class DynamicEvolution:
    def __init__(self, transformed, seed, specialist):
        self.seed = seed
        self.transformed = transformed
        self.data = transformed.data.copy()
        self.specialist = specialist
        self.specialist.predict = self.fake_predict

    @staticmethod
    def fake_predict():
        return 'predicted'

    def get_stage_data(self, index):
        data = self.data.query(f'index == {index}')
        self.transformed.set_data(data)

    def evolve_stage(self, stg):
        for trial in range(stg*10, (stg*10)+10):
            self.get_stage_data(trial)
            self.specialist.pre_process(self.transformed.X)
            self.specialist.post_process(self.transformed.y)
        return (
            self.specialist.generation,
            self.specialist.actual_score,
            bool(self.specialist.fit_start),
            bool(self.specialist.score_start)
        )

    def evolve_process(self, start, suffix='score'):
        results = []
        fits = []
        scores = []
        gens = []
        stages = len(self.data)//10
        for stg in range(start//10, stages):
            gen, result, fit_process, score_process = self.evolve_stage(stg)
            gens.append(gen)
            results.append(int(result*100))
            fits.append(fit_process)
            scores.append(score_process)

        df = pd.DataFrame({
            'generation': gens,
            'score': results,
            'score_process': scores,
            'fit_process': fits
        })
        df.to_csv(f'../../data/specialist/dynamic_evolution/{self.seed}_{suffix}.csv', index=False)
