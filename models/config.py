import yaml

class Config:
    def __init__(self, config_file):
        self.__build_config(config_file)
        self.__validate()

    @property
    def transform(self):
        return self.config.get('transform')
    
    @property
    def transform_type(self):
        return self.transform.get('type')

    @property
    def source(self):
        return self.config.get('source')
    
    @property
    def source_type(self):
        return self.source.get('type')


    def __build_config(self, config_file):
        with open(config_file, 'r') as stream:
            try:
                self.config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
    
    def __validate(self):
        expected = ['transform', 'source']
        for prop in expected:
            if prop not in self.config.keys():
                raise Exception(f'Missing property {prop} in config')
            if 'type' not in self.config[prop].keys():
                raise Exception(f'Property {prop} has no type associated')
