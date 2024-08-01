import pandas as pd
import numpy as np
from pathlib import Path
import os 
import sys
import pathlib

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(str(PACKAGE_ROOT))
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import config
from src.processing.data_handling import load_dataset,save_pipeline
import src.processing.preprocessing as pp
import src.pipeline as pipe

def perform_training():
    train_data = load_dataset(config.TRAIN_FILE)
    train_y = train_data[config.TARGET].map({'N':0,'Y':1})
    pipe.classification_pipeline.fit(train_data[config.FEATURES],train_y)
    save_pipeline(pipe.classification_pipeline)

if __name__ == '__main__':
    perform_training()
