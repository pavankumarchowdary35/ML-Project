import os
import pandas as pd
import joblib

from pathlib import Path
import sys

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent.parent
sys.path.append(str(PACKAGE_ROOT))

from src.config import config


## load the dataset
def load_dataset(file_name):
    filepath = os.path.join(config.DATAPATH, file_name)
    _data = pd.read_csv(filepath)
    return _data


## Serialization/model saving along with the pipeline
def save_pipeline(pipeline_to_save):
    save_path = os.path.join(config.SAVE_MODEL_PATH, config.MODEL_NAME)
    joblib.dump(pipeline_to_save, save_path)
    print(f"Model has been saved under the name {config.MODEL_NAME}")

## Deserialization/ model loading
def load_pipeline():
    save_path = os.path.join(config.SAVE_MODEL_PATH, config.MODEL_NAME)  
    model_loaded = joblib.load(save_path)  
    print(f"Model has been laoded")
    return model_loaded


