import os
import pandas as pd
from sklearn.model_selection import ShuffleSplit
from .problem import (  # noqa
#    get_cv,
    problem_title,
    prediction_type,
    workflow,
    prediction_labels,
    target_column_name,
)


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


def get_cv(X, y):
    cv = ShuffleSplit(n_splits=8, test_size=0.2, random_state=57)
    return cv.split(X)
