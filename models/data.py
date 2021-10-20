from models.transform.environmental import Environmental
from models.extract import Extract
from models.config import Config

class DataManager:
    def __init__(self, config_file):
        self.config = Config(config_file)

    def extract(self):
        extract = Extract(self.config.source)
        sources = {
            'local': extract.local,
            'remote': extract.remote
        }
        extract_data = sources[self.config.source_type]
        self.raw = extract_data()
        return self.raw

    def transform(self):
        models = {
            'environmental': Environmental
        }
        Transform = models[self.config.transform_type]
        self.transform = Transform(
            self.raw,
            self.config.transform
        )
        return self.transform
