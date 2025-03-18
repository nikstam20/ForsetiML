import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
import numpy as np
from aif360.datasets import StandardDataset
from sklearn.ensemble import RandomForestClassifier
from Modules.DataAnalysis.GroupFairness.notions import (
    conditional_statistical_parity,
    conditional_use_accuracy_equality,
    equal_negative_predictive_value,
    equal_opportunity,
    overall_accuracy_equality,
    predictive_equality,
    statistical_parity,
    treatment_equality,
    balance_for_positive_class,
    equalized_odds,
    balance_for_negative_class,
    test_fairness,
    well_calibration,
)


def feature_importance_suppression(
    data_path: str,
    target_column: str,
    sensitive_column: str,
    test_split_percent: float = 20.0,
    model_params: dict = None,
    privileged_classes=None,
    unprivileged_classes=None,
    random_state: int = 42,
    importance_threshold: float = 0.1,
):
    """
    Apply feature importance suppression and train/evaluate a model on the original and modified datasets.

    Parameters:
    - data_path: str - Path to the input CSV dataset.
    - target_column: str - Name of the target column in the dataset.
    - sensitive_column: str - Name of the sensitive attribute to debias.
    - test_split_percent: float - Proportion of data to be used as test set (percentage).
    - model_params: dict - Parameters for the XGBoost model.
    - privileged_classes: list - List of privileged classes for the sensitive attribute.
    - unprivileged_classes: list - List of unprivileged classes for the sensitive attribute.
    - random_state: int - Random seed for reproducibility.
    - importance_threshold: float - Threshold for feature importance below which features are removed.

    Returns:
    - metrics: dict - Dictionary containing model performance metrics before and after modification.
    """
    if model_params is None:
        model_params = {}
    if privileged_classes is None:
        raise ValueError(
            "privileged_classes and unprivileged_classes must be provided."
        )
    privileged_classes = [[1]]

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")
    data = pd.read_csv(data_path)

    if target_column not in data.columns or sensitive_column not in data.columns:
        raise ValueError(
            f"Target column '{target_column}' or sensitive column '{sensitive_column}' not found in data."
        )
    split_index = int(len(data) * ((1 - test_split_percent / 100)))
    train_data = data.iloc[:split_index]
    test_data = data.iloc[split_index:]

    X_train, y_train = (
        train_data.drop(columns=[target_column]),
        train_data[target_column],
    )
    X_test, y_test = test_data.drop(columns=[target_column]), test_data[target_column]
    target_encoder = LabelEncoder()
    y_train = target_encoder.fit_transform(y_train)
    y_test = target_encoder.transform(y_test)

    feature_encoders = {}
    categorical_cols = X_train.select_dtypes(include=["object"]).columns
    for col in categorical_cols:
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col])
        X_test[col] = le.transform(X_test[col])
        feature_encoders[col] = le

    train_ds = StandardDataset(
        df=pd.concat([X_train, pd.Series(y_train, name=target_column)], axis=1),
        label_name=target_column,
        favorable_classes=[1],
        protected_attribute_names=[sensitive_column],
        privileged_classes=privileged_classes,
        features_to_keep=X_train.columns.tolist(),
    )

    forest = RandomForestClassifier(random_state=random_state)
    forest.fit(X_train, y_train)
    importances = forest.feature_importances_
    important_features = [
        X_train.columns[i]
        for i in range(len(importances))
        if importances[i] < importance_threshold
    ]
    X_train_filtered = X_train[important_features]
    X_test_filtered = X_test[important_features]

    model_original = XGBClassifier(random_state=random_state, **model_params)
    model_filtered = XGBClassifier(random_state=random_state, **model_params)
    model_original.fit(X_train, y_train)
    y_pred_original = model_original.predict(X_test)
    model_filtered.fit(X_train_filtered, y_train)
    y_pred_filtered = model_filtered.predict(X_test_filtered)
    probabilities = model_filtered.predict_proba(X_test_filtered)
    accuracy_filtered = accuracy_score(y_test, y_pred_filtered)
    cm_filtered = confusion_matrix(y_test, y_pred_filtered)
    tn, fp, fn, tp = cm_filtered.ravel()

    sensitive_column_column = test_data[sensitive_column]

    y_test_with_sensitive = pd.DataFrame(
        {"y_test": y_test, f"{sensitive_column}": sensitive_column_column}
    )

    predictions_with_sensitive = pd.DataFrame(
        {"predictions": y_pred_filtered, f"{sensitive_column}": sensitive_column_column}
    )
    conditional_stat_parity = conditional_statistical_parity(
        y_test_with_sensitive["y_test"],
        predictions_with_sensitive["predictions"],
        y_test_with_sensitive[sensitive_column],
    )

    conditional_accuracy_eq = conditional_use_accuracy_equality(
        y_test_with_sensitive["y_test"],
        predictions_with_sensitive["predictions"],
        y_test_with_sensitive[sensitive_column],
    )

    equal_neg_predictive_val = equal_negative_predictive_value(
        y_test_with_sensitive["y_test"],
        predictions_with_sensitive["predictions"],
        y_test_with_sensitive[sensitive_column],
    )

    equal_opportunity_metric = equal_opportunity(
        y_test_with_sensitive["y_test"],
        predictions_with_sensitive["predictions"],
        y_test_with_sensitive[sensitive_column],
    )

    overall_accuracy_eq = overall_accuracy_equality(
        y_test_with_sensitive["y_test"],
        predictions_with_sensitive["predictions"],
        y_test_with_sensitive[sensitive_column],
    )

    predictive_eq = predictive_equality(
        y_test_with_sensitive["y_test"],
        predictions_with_sensitive["predictions"],
        y_test_with_sensitive[sensitive_column],
    )

    statistical_parity_metric = statistical_parity(
        y_test_with_sensitive["y_test"],
        predictions_with_sensitive["predictions"],
        predictions_with_sensitive[sensitive_column],
    )

    treatment_eq = treatment_equality(
        y_test_with_sensitive["y_test"],
        predictions_with_sensitive["predictions"],
        y_test_with_sensitive[sensitive_column],
    )

    balance_positive_class = balance_for_positive_class(
        y_test_with_sensitive["y_test"],
        predictions_with_sensitive["predictions"],
        y_test_with_sensitive[sensitive_column],
    )

    equalized_odds_metric = equalized_odds(
        y_test_with_sensitive["y_test"],
        predictions_with_sensitive["predictions"],
        y_test_with_sensitive[sensitive_column],
    )

    balance_negative_class = balance_for_negative_class(
        y_test_with_sensitive["y_test"],
        predictions_with_sensitive["predictions"],
        y_test_with_sensitive[sensitive_column],
    )

    test_fairness_metric = test_fairness(
        y_test_with_sensitive["y_test"],
        predictions_with_sensitive["predictions"],
        y_test_with_sensitive[sensitive_column],
        probabilities,
    )

    well_calibration_metric = well_calibration(
        y_test_with_sensitive["y_test"],
        predictions_with_sensitive["predictions"],
        y_test_with_sensitive[sensitive_column],
        probabilities,
    )

    precision_repaired = precision_score(y_test, y_pred_filtered)
    recall_repaired = recall_score(y_test, y_pred_filtered)
    f1_score_repaired = f1_score(y_test, y_pred_filtered)

    altered_features = X_train.shape[1] - X_train_filtered.shape[1]
    total_features = X_train.shape[1]
    alteration_ratio = altered_features / total_features

    metrics = {
        "predictions": y_pred_filtered.tolist(),
        "probabilities": probabilities.tolist(),
        "accuracy": float(accuracy_filtered),
        "precision": float(precision_repaired),
        "recall": float(recall_repaired),
        "f1_score": float(f1_score_repaired),
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        },
        "conditional_statistical_parity": conditional_stat_parity,
        "conditional_use_accuracy_equality": conditional_accuracy_eq,
        "equal_negative_predictive_value": equal_neg_predictive_val,
        "equal_opportunity": equal_opportunity_metric,
        "overall_accuracy_equality": overall_accuracy_eq,
        "predictive_equality": predictive_eq,
        "statistical_parity": statistical_parity_metric,
        "treatment_equality": treatment_eq,
        "balance_for_positive_class": balance_positive_class,
        "equalized_odds": equalized_odds_metric,
        "balance_for_negative_class": balance_negative_class,
        "test_fairness": test_fairness_metric,
        "well_calibration": well_calibration_metric,
        "alteration_ratio": alteration_ratio,
    }
    return metrics
