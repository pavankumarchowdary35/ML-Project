import sys
import os 
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(str(PACKAGE_ROOT))

from src.config import config
from src.processing.data_handling import load_dataset, load_pipeline

classification_pipeline = load_pipeline()


def generate_predictions(data_input):
    data = pd.DataFrame(data_input)
    pred = classification_pipeline.predict(data[config.FEATURES])
    output = np.where(pred==1,'Y','N')
    result = {"prediction":output}
    return result

def gen_predict():
    test_data = load_dataset(config.TEST_FILE)
    pred = classification_pipeline.predict(test_data[config.FEATURES])
    output = np.where(pred==1,'Y','N')
    print(output)
    result = {'prediction':output}
    return result


if __name__ == '__main__':
    generate_predictions()