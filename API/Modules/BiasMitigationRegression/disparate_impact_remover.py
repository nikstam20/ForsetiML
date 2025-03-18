import os
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from aif360.datasets import RegressionDataset
from aif360.algorithms.preprocessing import DisparateImpactRemover
from Modules.DataAnalysis.GroupFairness.notions_regression import total


def dir_regression(
    data_path: str,
    target_column: str,
    sensitive_column: str,
    repair_level: float = 1.0,
    test_split_percent: float = 20.0,
    model_params: dict = None,
    random_state: int = 42,
):
    """
    Apply Disparate Impact Remover and train/evaluate a regression model using XGBoost.

    Parameters:
    - data_path: str
        Path to the input CSV dataset.
    - target_column: str
        Name of the target column in the dataset.
    - sensitive_column: str
        Name of the sensitive attribute to debias.
    - repair_level: float
        Level of repair (0.0 = no repair, 1.0 = full repair).
    - test_split_percent: float
        Proportion of data to be used as test set (percentage).
    - model_params: dict
        Parameters for the XGBoost model.
    - random_state: int
        Random seed for reproducibility.

    Returns:
    - metrics: dict
        Dictionary containing model performance metrics before and after repair.
    """
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

    privileged_classes = [1]
    unprivileged_classes = [0]

    train_ds = RegressionDataset(
        df=X_train.join(y_train),
        dep_var_name=target_column,
        protected_attribute_names=[sensitive_column],
        privileged_classes=[privileged_classes],
    )

    test_ds = RegressionDataset(
        df=X_test.join(y_test),
        dep_var_name=target_column,
        protected_attribute_names=[sensitive_column],
        privileged_classes=[privileged_classes],
    )

    di_remover = DisparateImpactRemover(
        repair_level=repair_level, sensitive_attribute=sensitive_column
    )
    train_ds_repaired = di_remover.fit_transform(train_ds)

    X_train_repaired = pd.DataFrame(train_ds_repaired.features, columns=X_train.columns)
    y_train_repaired = train_ds_repaired.labels.ravel()

    model = XGBRegressor(random_state=random_state, **model_params)
    model_repaired = XGBRegressor(random_state=random_state, **model_params)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    model_repaired.fit(X_train_repaired, y_train_repaired)
    y_pred_repaired = model_repaired.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mse_repaired = mean_squared_error(y_test, y_pred_repaired)
    mae_repaired = mean_absolute_error(y_test, y_pred_repaired)
    r2_repaired = abs(r2_score(y_test, y_pred_repaired))

    sensitive_feature_column = X_test[sensitive_column]
    y_test_with_sensitive = pd.DataFrame(
        {"y_test": y_test, sensitive_column: sensitive_feature_column}
    ).reset_index(drop=True)
    predictions_with_sensitive = pd.DataFrame(
        {"predictions": y_pred_repaired, sensitive_column: sensitive_feature_column}
    ).reset_index(drop=True)
    findexes = total(
        y_test_with_sensitive, predictions_with_sensitive, sensitive_column
    )
    metrics = {
        "mse": mse_repaired,
        "mae": mae_repaired,
        "r2_score": r2_repaired,
    }
    epsilon = 1e-8
    feature_shifts = np.abs(X_train.values - X_train_repaired) / (
        np.abs(X_train.values) + epsilon
    )
    alteration_ratio = np.mean(feature_shifts)
    metrics.update(findexes)
    metrics["alteration_ratio"] = float(alteration_ratio)
    for key, value in metrics.items():
        if isinstance(value, np.floating):
            metrics[key] = float(value)
    return metrics
