from sklearn.ensemble import RandomForestRegressor


def get_estimator():
    pipeline = RandomForestRegressor(
        n_estimators=50, max_leaf_nodes=3, random_state=61)
    return pipeline
