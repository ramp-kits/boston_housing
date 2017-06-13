"""Script called on the backend to prepare data files.

Typically, it creates data/public_train.csv and data/public_test.csv
which will be committed into the repo and used by the starting kit,
and data/train.csv and data/test.csv which are kept locally.
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split


df = pd.read_csv(os.path.join('data', 'boston_housing.csv'), index_col=0)
df_train, df_test = train_test_split(
    df, test_size=0.2, random_state=57)
df_train.to_csv(os.path.join('data', 'train.csv'), index=False)
df_test.to_csv(os.path.join('data', 'test.csv'), index=False)

# It is a good pracice to make the public data independent of both
# the training and test data on the backend, but it is also fine
# if the public data is the same as the training data (e.g., in case
# we don't have much data to spare), since "cheaters"
# can be caught by looking at their code and by them overfitting the
# public leaderboard.
df_public = df_train
df_public_train, df_public_test = train_test_split(
    df_public, test_size=0.2, random_state=57)
df_public_train.to_csv(os.path.join('data', 'public_train.csv'), index=False)
df_public_test.to_csv(os.path.join('data', 'public_test.csv'), index=False)
