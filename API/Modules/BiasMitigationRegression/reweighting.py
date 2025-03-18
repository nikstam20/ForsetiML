import os
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from Modules.DataAnalysis.GroupFairness.notions_regression import total


def reweighting_regression(
    data_path: str,
    target_column: str,
    sensitive_column: str,
    test_split_percent: float = 20.0,
    model_params: dict = None,
    random_state: int = 42,
):
    if model_params is None:
        model_params = {}
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    data = pd.read_csv(data_path)
    data[sensitive_column] = LabelEncoder().fit_transform(data[sensitive_column])

    X = data.drop(columns=[target_column])
    y = data[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_split_percent / 100, random_state=random_state
    )

    sensitive_train = X_train[sensitive_column]
    group_counts = sensitive_train.value_counts(normalize=True)
    group_weights = {
        group: 1.0 / proportion for group, proportion in group_counts.items()
    }
    sample_weights = sensitive_train.map(group_weights).values

    model = XGBRegressor(random_state=random_state, **model_params)
    model.fit(X_train, y_train, sample_weight=sample_weights)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    sensitive_feature_column = X_test[sensitive_column]
    findexes = total(
        pd.DataFrame(
            {"y_test": y_test, sensitive_column: sensitive_feature_column}
        ).reset_index(drop=True),
        pd.DataFrame(
            {"predictions": y_pred, sensitive_column: sensitive_feature_column}
        ).reset_index(drop=True),
        sensitive_column,
    )

    alteration_ratio = 0.0

    metrics = {
        "mse": mse,
        "mae": mae,
        "r2_score": r2,
        "alteration_ratio": alteration_ratio,
    }
    metrics.update(findexes)
    for key, value in metrics.items():
        if isinstance(value, np.floating):
            metrics[key] = float(value)

    return metrics
