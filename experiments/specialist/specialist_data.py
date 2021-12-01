# Configuration for running Jupyter Notebooks with project models
import sys
import os
import pandas as pd

module_path = os.path.abspath(os.path.join(os.pardir, os.pardir))
if module_path not in sys.path:
    sys.path.append(module_path)

# Transforming data
from models.data import DataManager
from sklearn.model_selection import train_test_split

manager = DataManager('../../configs/xdpole.yml')
manager.extract()
transformed = manager.transform()

def level(value):
    if value == 1000:
        return 'good'
    else:
        return 'bad'

transformed.create_level_label(level_func=level)

def get_stage(stage):
    stages = {
        'lifetime': transformed.data,
        'birth': transformed.data.query('index < 7500'),
        'growth': transformed.data.query('index > 7500 and index < 15000'),
        'young': transformed.data.query('index < 15000'),
        'adult': transformed.data.query('index > 12000 and index < 30000'),
        'old': transformed.data.query('index > 20000 and index < 35000'),
        'elder': transformed.data.query('index > 30000'),
        'mature': transformed.data.query('index > 9000'),
    }
    data = stages.get(stage)
    transformed.set_data(data)
    return train_test_split(transformed.X, transformed.level, test_size=0.33, random_state=42)
