import os
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from Modules.DataAnalysis.GroupFairness.notions_regression import total


def feature_removal_regression(
    data_path: str,
    target_column: str,
    sensitive_column: str,
    test_split_percent: float = 20.0,
    model_params: dict = None,
    random_state: int = 42,
):
    """
    Removes the sensitive feature and trains a regression model using XGBoost.

    Parameters:
    - data_path: str
    - target_column: str
    - sensitive_column: str
    - test_split_percent: float
    - model_params: dict
    - random_state: int

    Returns:
    - metrics: dict
    """
    if model_params is None:
        model_params = {}
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    data = pd.read_csv(data_path)
    X = data.drop(columns=[target_column])
    y = data[target_column]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_split_percent / 100, random_state=random_state
    )
    sensitive_feature_column = X_test[sensitive_column]

    data = data.drop(columns=[sensitive_column])

    X = data.drop(columns=[target_column])
    y = data[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_split_percent / 100, random_state=random_state
    )
    model = XGBRegressor(random_state=random_state, **model_params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    findexes = total(
        pd.DataFrame(
            {"y_test": y_test, sensitive_column: sensitive_feature_column}
        ).reset_index(drop=True),
        pd.DataFrame(
            {"predictions": y_pred, sensitive_column: sensitive_feature_column}
        ).reset_index(drop=True),
        sensitive_column,
    )
    metrics = {
        "mse": mse,
        "mae": mae,
        "r2_score": r2,
    }
    metrics.update(findexes)
    for key, value in metrics.items():
        if isinstance(value, np.floating):
            metrics[key] = float(value)
    return metrics
