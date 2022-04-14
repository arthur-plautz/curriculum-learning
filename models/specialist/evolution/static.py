import os
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

class StaticEvolution:
    def __init__(self, transformed, seed, specialist, batch_size, trials=10):
        self.seed = seed
        self.transformed = transformed
        self.trials = 10
        self.data = transformed.data.copy()
        self.specialist = specialist
        self.batch_size = batch_size
        self.specialist.fit_batch_size = self.batch_size
        self.specialist.score_batch_size = self.batch_size
        self.save_path = f'../../data/specialist/static_evolution/{self.specialist.model.type}'
        self.verify_dir()

    def verify_dir(self):
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)

    @staticmethod
    def fake_predict():
        return 'predicted'

    def get_stage_data(self, stg):
        data = self.data.query(f'index >= {stg*self.trials} and index < {(stg+1)*self.trials}')
        self.transformed.set_data(data)

    def evolve_stage(self, stg):
        if stg >= self.specialist.start_generation:
            self.get_stage_data(stg)
            self.specialist.evaluation(self.transformed.X_list, self.transformed.performance_list)
            result = (
                self.specialist.generation,
                self.specialist.actual_score
            )
            return result

    def evolve_process(self, start, suffix='score'):
        results = []
        gens = []
        stages = len(self.data)//self.trials
        for stg in range(start, stages):
            stage_result = self.evolve_stage(stg)
            if stage_result:
                gen, result = stage_result
                gens.append(gen)
                results.append(int(result*100))
        df = pd.DataFrame({
            'stages': gens,
            'score': results
        })
        df.to_csv(f'{self.save_path}/{int(self.batch_size)}_bs_{self.seed}_{suffix}.csv', index=False)
