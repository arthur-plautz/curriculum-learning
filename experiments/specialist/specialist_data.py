# Configuration for running Jupyter Notebooks with project models
import sys
import os
import pandas as pd

module_path = os.path.abspath(os.path.join(os.pardir, os.pardir))
if module_path not in sys.path:
    sys.path.append(module_path)

# Transforming data
from models.data import DataManager

def level(value):
    if value == 1000:
        return 'good'
    else:
        return 'bad'

manager = DataManager('../../configs/xdpole.yml')
manager.extract()
manager.level_function(level)
manager.transform_all()

# Bringing the Specialist
from models.specialist import Specialist
specialist = Specialist(manager.s10_transformed, 10)
specialist.evolve_process(interval=0.05)
