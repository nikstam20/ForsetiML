import os
import time
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from Modules.DataAnalysis.GroupFairness.notions_regression import total


def prejudice_remover_regression(
    data_path: str,
    target_column: str,
    sensitive_column: str,
    test_split_percent: float = 20.0,
    fairness_penalty: float = 0.1,
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

    X_train_no_sensitive = X_train.drop(columns=[sensitive_column])
    X_test_no_sensitive = X_test.drop(columns=[sensitive_column])

    model = XGBRegressor(random_state=random_state, **model_params)

    start_time = time.time()
    model.fit(X_train_no_sensitive, y_train)
    y_pred = model.predict(X_test_no_sensitive)
    fairness_violation = np.abs(np.corrcoef(y_pred, X_test[sensitive_column])[0, 1])

    sample_weights = np.ones(len(y_train))
    for _ in range(5):
        model.fit(X_train_no_sensitive, y_train, sample_weight=sample_weights)
        y_pred = model.predict(X_test_no_sensitive)
        fairness_violation = np.abs(np.corrcoef(y_pred, X_test[sensitive_column])[0, 1])
        sample_weights = np.exp(
            fairness_penalty
            * np.abs(X_train[sensitive_column] - np.mean(X_train[sensitive_column]))
        )

    total_time = time.time() - start_time
    model_file = "prejudice_remover_model.pkl"
    joblib.dump(model, model_file)
    model_size = os.path.getsize(model_file)

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

    metrics = {
        "mse": mse,
        "mae": mae,
        "r2_score": r2,
        "training_time": total_time,
        "model_size": model_size,
    }
    metrics.update(findexes)

    for key, value in metrics.items():
        if isinstance(value, np.floating):
            metrics[key] = float(value)

    return metrics
