import os
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from aif360.datasets import RegressionDataset
from aif360.algorithms.preprocessing import PrevalenceSampling
from Modules.DataAnalysis.GroupFairness.notions_regression import total


def prevalence_sampling_regression(
    data_path,
    target_column,
    sensitive_column,
    test_split_percent=20.0,
    model_params=None,
    random_state=42,
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

    train_ds = RegressionDataset(
        df=X_train.join(y_train),
        dep_var_name=target_column,
        protected_attribute_names=[sensitive_column],
    )

    prevalence_sampler = PrevalenceSampling(
        unprivileged_groups=[{sensitive_column: 0}],
        privileged_groups=[{sensitive_column: 1}],
    )

    train_ds_repaired = prevalence_sampler.fit_transform(train_ds)

    X_train_repaired = pd.DataFrame(train_ds_repaired.features, columns=X_train.columns)
    y_train_repaired = train_ds_repaired.labels.ravel()

    model = XGBRegressor(random_state=random_state, **model_params)
    model_repaired = XGBRegressor(random_state=random_state, **model_params)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    model_repaired.fit(X_train_repaired, y_train_repaired)
    y_pred_repaired = model_repaired.predict(X_test)

    mse_repaired = mean_squared_error(y_test, y_pred_repaired)
    mae_repaired = mean_absolute_error(y_test, y_pred_repaired)
    r2_repaired = r2_score(y_test, y_pred_repaired)

    sensitive_feature_column = X_test[sensitive_column]
    findexes = total(
        pd.DataFrame({"y_test": y_test, sensitive_column: sensitive_feature_column}),
        pd.DataFrame(
            {"predictions": y_pred_repaired, sensitive_column: sensitive_feature_column}
        ),
        sensitive_column,
    )

    alteration_ratio = (
        np.mean(
            np.abs(X_train.values - X_train_repaired) / (np.abs(X_train.values) + 1e-8)
        )
        * 100
    )

    metrics = {
        "mse": mse_repaired,
        "mae": mae_repaired,
        "r2_score": r2_repaired,
        "alteration_ratio (%)": float(alteration_ratio),
    }
    metrics.update(findexes)

    return metrics
