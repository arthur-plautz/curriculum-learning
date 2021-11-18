from models.stats.seed_stats import SeedStats
import matplotlib.pyplot as plt

class ContextStats:
    def __init__(self, data_dir, seeds):
        self.data_dir = data_dir
        self.seeds = {}
        self.add_seeds(seeds)

    def get_seed(self, seed):
        return self.seeds.get(seed)

    def add_seeds(self, seeds):
        for seed in seeds:
            self.add_seed(seed)

    def add_seed(self, seed):
        self.seeds[seed] = SeedStats(self.data_dir, seed)

    def fitness_evolution(self):
        for seed in self.seeds.values():
            seed.fitness_evolution(show=False)
        plt.show()

    def fitness_evolution_boxplot(self):
        avgs = [s.run_data.avgfit for s in self.seeds.values()]
        plt.boxplot(avgs)
        plt.show()
