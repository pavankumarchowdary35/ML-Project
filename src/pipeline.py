from sklearn.pipeline import Pipeline
from pathlib import Path
import os
import sys

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(str(PACKAGE_ROOT))

from src.config import config
import src.processing.preprocessing as pp
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
import numpy as np


classification_pipeline = Pipeline(
[ 
    ('DomainProcessing', pp.DomainProcessing(variable_to_modify=config.FEATURES_TO_MODIFY, variable_to_add= config.FEATURES_TO_ADD)),
    ('MeanImputation', pp.MeanImputer(variables=config.NUM_FEATURES)),
    ('ModeImputation', pp.ModeImputer(variables= config.CAT_FEATURES)),
    ('DropFeatures', pp.DropColumns(variables_to_drop=config.DROP_FEATURES)),
    ('LabelEncode', pp.CustomLabelEncoder(variables=config.FEATURES_TO_ENCODE)),
    ('LogTransform',pp.LogTransforms(variables=config.LOG_FEATURES)),
    ('MinMaxscale', MinMaxScaler()),
    ('LogisticClassifier', LogisticRegression(random_state=0))
]
)