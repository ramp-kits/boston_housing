import os
import imp
import pandas as pd
from sklearn.model_selection import ShuffleSplit

problem = imp.load_source('', 'problem.py')
problem_title = problem.problem_title
prediction_type = problem.prediction_type
prediction_labels = problem.prediction_labels
workflow = problem.workflow
target_column_name = problem.target_column_name
# You can use the starting kit CV or redefine the get_cv function
# get_cv = problem.get_cv


def get_cv(X, y):
    cv = ShuffleSplit(n_splits=8, test_size=0.2, random_state=57)
    return cv.split(X)


def prepare_data():
    pass


def get_train_data(path='.'):
    data = pd.read_csv(os.path.join(path, 'private_data', 'train.csv'))
    y_array = data[target_column_name].values
    X_array = data.drop([target_column_name], axis=1).values
    return X_array, y_array


def get_test_data(path='.'):
    data = pd.read_csv(os.path.join(path, 'private_data', 'test.csv'))
    y_array = data[target_column_name].values
    X_array = data.drop([target_column_name], axis=1).values
    return X_array, y_array
