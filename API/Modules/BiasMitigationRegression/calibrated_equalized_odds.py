import os
import time
import joblib
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from Modules.DataAnalysis.GroupFairness.notions_regression import total


def calibrated_equalized_odds_regression(
    data_path: str,
    target_column: str,
    sensitive_column: str,
    test_split_percent: float = 20.0,
    calibration_strength: float = 0.1,
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

    model = XGBRegressor(random_state=random_state, **model_params)

    start_time = time.time()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    group_means = y_test.groupby(X_test[sensitive_column]).mean()
    group_residuals = X_test[sensitive_column].map(
        lambda g: group_means[g] - np.mean(y_test)
    )

    y_pred_adjusted = y_pred + calibration_strength * group_residuals

    data_distortion = (
        np.mean(np.abs(y_pred_adjusted - y_pred) / (np.abs(y_pred) + 1e-8)) * 100
    )

    total_time = time.time() - start_time
    model_file = "calibrated_equalized_odds_model.pkl"
    joblib.dump(model, model_file)
    model_size = os.path.getsize(model_file)

    mse = mean_squared_error(y_test, y_pred_adjusted)
    mae = mean_absolute_error(y_test, y_pred_adjusted)
    r2 = r2_score(y_test, y_pred_adjusted)

    sensitive_feature_column = X_test[sensitive_column]
    findexes = total(
        pd.DataFrame(
            {"y_test": y_test, sensitive_column: sensitive_feature_column}
        ).reset_index(drop=True),
        pd.DataFrame(
            {"predictions": y_pred_adjusted, sensitive_column: sensitive_feature_column}
        ).reset_index(drop=True),
        sensitive_column,
    )
    metrics = {
        "mse": mse,
        "mae": mae,
        "r2_score": r2,
        "training_time_seconds": total_time,
        "model_size_bytes": model_size,
        "data_distortion": data_distortion,
    }
    metrics.update(findexes)

    for key, value in metrics.items():
        if isinstance(value, np.floating):
            metrics[key] = float(value)

    return metrics
