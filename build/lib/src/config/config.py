import pathlib
import os
import src


PACKAGE_ROOT = pathlib.Path(__file__).resolve().parent.parent

DATAPATH = os.path.join(PACKAGE_ROOT,'datasets')

TRAIN_FILE = 'train.csv'
TEST_FILE = 'test.csv'

MODEL_NAME = 'classification.pkl'
SAVE_MODEL_PATH = os.path.join(PACKAGE_ROOT,'trained_models')

TARGET = 'Loan_Status'

FEATURES = ['Gender','Married','Dependents','Education','Self_Employed','ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History','Property_Area']

NUM_FEATURES = ['ApplicantIncome','LoanAmount', 'Loan_Amount_Term']

CAT_FEATURES = ['Gender','Married','Dependents','Education','Self_Employed', 'Credit_History','Property_Area']

FEATURES_TO_ENCODE = ['Gender','Married','Dependents','Education','Self_Employed', 'Credit_History','Property_Area']

FEATURES_TO_ADD = 'CoapplicantIncome'
FEATURES_TO_MODIFY = ['ApplicantIncome']

DROP_FEATURES = ['CoapplicantIncome']

LOG_FEATURES = ['ApplicantIncome', 'LoanAmount'] # taking log of numerical columns