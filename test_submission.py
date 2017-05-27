#!/usr/bin/env
from __future__ import print_function
import problem
print("Reading file ...")
X, y = problem.get_train_data()
print("Training model ...")
cv = problem.get_cv(y)
module_path = 'submissions.starting_kit'
for train_is, test_is in cv:
    trained_workflow = problem.workflow.train_submission(
        module_path, X, y, train_is=train_is)
    y_pred = problem.workflow.test_submission(trained_workflow, X)
    predictions = problem.prediction.Predictions(y_pred=y_pred[test_is])
    ground_truth = problem.prediction.Predictions(y_true=y[test_is])
    for score_type in problem.score_types:
        score = score_type.score_function(ground_truth, predictions)
        print('{} = {}'.format(score_type.name, score))
