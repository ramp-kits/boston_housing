import os
import pandas as pd
import rampwf as rw
from sklearn.model_selection import ShuffleSplit

problem_title = 'Boston housing price regression'
prediction_type = rw.prediction_types.regression
workflow = rw.workflows.Regressor()
prediction_labels = None

score_types = [
    rw.score_types.RMSE(),
    rw.score_types.RelativeRMSE(name='rel_rmse'),
]


def get_data(path='.'):
    data = pd.read_csv(os.path.join(path, 'public_data', 'public_train.csv'))
    target_column_name = 'medv'
    y_array = data[target_column_name].values
    X_array = data.drop([target_column_name], axis=1).values
    return X_array, y_array


def get_cv(X, y):
    cv = ShuffleSplit(n_splits=2, test_size=0.2, random_state=57)
    return cv.split(X)
