import pytest
import os
import sys

from pathlib import Path


PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent
print('root is ',PACKAGE_ROOT)
sys.path.append(PACKAGE_ROOT)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from src.config import config
from src.processing.data_handling import load_dataset
from src.predict import generate_predictions


@pytest.fixture
def single_prediction():
    test_dataset = load_dataset(config.TEST_FILE)
    single_row = test_dataset[:1]
    result = generate_predictions(single_row)
    return result

def test_single_pred_not_none(single_prediction): # output is not none
    assert single_prediction is not None

def test_single_pred_str_type(single_prediction): # data type is string
    assert isinstance(single_prediction.get('prediction')[0],str)

def test_single_pred_validate(single_prediction): # check the output is Y
    assert single_prediction.get('prediction')[0] == 'Y'
