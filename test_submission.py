#!/usr/bin/env python
from __future__ import print_function
from problem import problem_title, get_train_data, get_cv, workflow,\
    prediction, score_types
print('Testing {}'.format(problem_title))
print('Reading file ...')
X, y = get_train_data()
print('Training model ...')
cv = get_cv(y)
module_path = 'submissions.starting_kit'
for train_is, test_is in cv:
    trained_workflow = workflow.train_submission(
        module_path, X, y, train_is=train_is)
    y_pred = workflow.test_submission(trained_workflow, X)
    predictions = prediction.Predictions(y_pred=y_pred[test_is])
    ground_truth = prediction.Predictions(y_true=y[test_is])
    for score_type in score_types:
        score = score_type.score_function(ground_truth, predictions)
        print('{} = {}'.format(score_type.name, score))
