from models.transform.environmental import Environmental
from models.extract import Extract
from models.config import Config

class DataManager:
    def __init__(self, config_file):
        self.config = Config(config_file)
        self.transformed = {}

    def extract(self):
        extract = Extract(self.config.source)
        sources = {
            'local': extract.local,
            'remote': extract.remote
        }
        extract_data = sources[self.config.source_type]
        self.raw = extract_data()
        return self.raw

    def level_function(self, level_func):
        self.__level_func = level_func

    def transform(self, seed):
        models = {
            'environmental': Environmental
        }
        Transform = models[self.config.transform_type]
        transformed = Transform(
            self.raw.get(seed),
            self.config.transform,
        )
        if self.__level_func:
            transformed.create_level_label(self.__level_func)
        return transformed

    def transform_all(self):
        for seed in self.raw.keys():
            transformed = self.transform(seed)
            setattr(self, seed+'_transformed', transformed)
            self.transformed[seed] = transformed

    def set_target(self, seed):
        self.target_data = self.transformed.get(seed)
