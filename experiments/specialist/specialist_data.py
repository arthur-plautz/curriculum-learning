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
transformed.create_level_label()

# split train/test data
X_train, X_test, y_train, y_test = train_test_split(transformed.X, transformed.level, test_size=0.33, random_state=42)
