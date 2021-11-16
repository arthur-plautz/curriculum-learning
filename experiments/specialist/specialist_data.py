# Configuration for running Jupyter Notebooks with project models
import sys
import os

module_path = os.path.abspath(os.path.join(os.pardir, os.pardir))
if module_path not in sys.path:
    sys.path.append(module_path)

# Transforming data
from models.data import DataManager
from models.stats import Stats

manager = DataManager('../../configs/xdpole.yml')
manager.extract()
transformed = manager.transform()
transformed.create_level_label()

stats = Stats('../../data/xdpole', 10)