from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import numpy as np

def evaluate_rf_params(args):
    """
    This function is used by multiprocessing Pool.
    args = (X_train, y_train, n_estimators, max_depth, random_state)
    """
    X_train, y_train, n_estimators, max_depth, random_state = args

    model = RandomForestClassifier(
    n_estimators=int(n_estimators),
    max_depth=int(max_depth),
    random_state=random_state,
    n_jobs=1
   )


    # 3-fold CV for fitness stability
    scores = cross_val_score(model, X_train, y_train, cv=3, scoring="accuracy")
    return float(np.mean(scores))
