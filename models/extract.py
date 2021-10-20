import pandas as pd

class Extract:
    def __init__(self, config):
        self.config = config
    
    def local(self):
        sources = self.config.get('files')
        dfs = []
        for source in sources:
            df = pd.read_csv(source, index_col=False)
            dfs.append(df)
        return pd.concat(dfs, ignore_index=True).reset_index(drop=True)

    def remote(self):
        pass
