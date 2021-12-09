import pandas as pd

class Extract:
    def __init__(self, config):
        self.config = config
    
    def local(self):
        sources = self.config.get('files')
        seeds = {}
        for source in sources:
            if isinstance(source, dict):
                [seed] = source.keys()
                [source_file] = source.values()
                df = pd.read_csv(source_file, index_col=False)
                seeds[seed] = df
            else:
                raise Exception('Files config must be a dict')            
        return seeds

    def remote(self):
        pass
