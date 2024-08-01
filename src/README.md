# Packaging the ML Model of Classification

#### Problem Statement
- Company wants to automate the loan eligibility process based on customer detail provided while filling online application form. 
- It is a classification problem where we have to predict whether a loan would be approved or not. 

#### Data
The data corresponds to a set of financial requests associated with individuals. 

| Variables         | Description                                    |
|-------------------|------------------------------------------------|
| Loan_ID           | Unique Loan ID                                 |
| Gender            | Male/ Female                                   |
| Married           | Applicant married (Y/N)                        |
| Dependents        | Number of dependents                           |
| Education         | Applicant Education (Graduate/ Under Graduate) |
| Self_Employed     | Self employed (Y/N)                            |
| ApplicantIncome   | Applicant income                               |
| CoapplicantIncome | Coapplicant income                             |
| LoanAmount        | Loan amount in thousands                       |
| Loan_Amount_Term  | Term of loan in months                         |
| Credit_History    | credit history meets guidelines                |
| Property_Area     | Urban/ Semi Urban/ Rural                       |
| Loan_Status       | Loan approved (Y/N)                            |

Source: Kaggle


## Virtual Environment
Install virtualenv

```python
python3 -m pip install virtualenv
```

Check version
```python
virtualenv --version
```

Create virtual environment

```python
virtualenv venv
```

Activate virtual environment

For Linux/Mac
```python
source venv/bin/activate
```
For Windows
```python
venv\Scripts\activate
```

Deactivate virtual environment

```python
deactivate
```


## Directory structure

```bash
src


├── MANIFEST.in
├── src
│   ├── config
│   │   ├── config.py
│   │   └── __init__.py
│   ├── datasets
│   │   ├── __init__.py
│   │   ├── test.csv
│   │   └── train.csv
│   ├── __init__.py
│   ├── pipeline.py
│   ├── predict.py
│   ├── processing
│   │   ├── data_handling.py
│   │   ├── __init__.py
│   │   └── preprocessing.py
│   ├── trained_models
│   │   ├── classification.pkl
│   │   └── __init__.py
│   ├── training_pipeline.py
│   └── VERSION
├── README.md
├── requirements.txt
├── setup.py
└── tests
    ├── pytest.ini
    └── test_prediction.py
```


# Build the Package

1. Goto Project directory and install dependencies
`pip install -r requirements.txt`

2. Create Pickle file after training:
`python src/training_pipeline.py`

3. Create source distribution and wheel
`python setup.py sdist bdist_wheel`

# Installation of Package

Go to project directory where `setup.py` file is located

1. To install it in editable or developer mode
```python
pip install -e .
```
```.``` refers to current directory

```-e``` refers to --editable mode

2. Normal installation
```python
pip install .
```
```.``` refers to current directory

3. Also can be installed from git as well after pushing to github

```
pip install git+https://github.com/pavankumarchowdary35/ML-Project.git
```

# Testing the Package Working

1. Create a new virual environment using the commands mentioned above & activate it
2. Now in the new environment install the package from github
`pip install git+https://github.com/pavankumarchowdary35/ML-Project.git`
3. Now to make prediction on your custom data , run the following code. Make sure that your custom data should have the features described above.
```
from src import training_pipeline 
import pandas as pd
from src import predict

training_pipeline.perform_training()
test_data = pd.read_csv('test.csv')

predict.generate_predictions(test_data[:1])

```



