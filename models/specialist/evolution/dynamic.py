import os
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

class DynamicEvolution:
    def __init__(self, transformed, seed, specialist, trials=10):
        self.seed = seed
        self.transformed = transformed
        self.data = transformed.data.copy()
        self.specialist = specialist
        self.specialist.predict = self.fake_predict
        self.trials = trials
        self.save_path = f'../../data/specialist/dynamic_evolution/{self.specialist.model.type}'
        self.verify_dir()

    @staticmethod
    def fake_predict():
        return 'predicted'

    def get_stage_data(self, stg):
        data = self.data.query(f'index >= {stg} and index < {stg+self.trials}')
        self.transformed.set_data(data)

    def evolve_stage(self, stg):
        if stg >= self.specialist.start_generation:
            self.get_stage_data(stg)
            self.specialist.evaluation(self.transformed.X_normalized, self.transformed.y)
            return (
                self.specialist.generation,
                self.specialist.actual_score,
                'fit' if self.specialist.fit_start else 'score'
            )

    def verify_dir(self):
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)

    def evolve_process(self, start, suffix='score'):
        results = []
        cycles = []
        gens = []
        stages = len(self.data)//self.trials
        for stg in range(start, stages):
            stage_result = self.evolve_stage(stg)
            if stage_result:
                gen, result, cycle = stage_result
                gens.append(gen)
                results.append(int(result*100))
                cycles.append(cycle)
        df = pd.DataFrame({
            'generation': gens,
            'score': results,
            'cycle': cycles
        })
        df.to_csv(f'{self.save_path}/{self.seed}_{suffix}.csv', index=False)
