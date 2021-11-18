# Configuration for running Jupyter Notebooks with project models
import sys
import os
import pandas as pd

module_path = os.path.abspath(os.path.join(os.pardir, os.pardir))
if module_path not in sys.path:
    sys.path.append(module_path)

# Transforming data
from models.data import DataManager

manager = DataManager('../../configs/xdpole.yml')
manager.extract()
transformed = manager.transform()
transformed.create_level_label()
