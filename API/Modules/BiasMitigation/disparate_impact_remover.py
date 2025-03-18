import os
import pickle
import pandas as pd
from aif360.algorithms.preprocessing import DisparateImpactRemover
from aif360.datasets import StandardDataset
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
from Modules.DataAnalysis.GroupFairness.notions import (
    conditional_statistical_parity,
    conditional_use_accuracy_equality,
    equal_negative_predictive_value,
    equal_opportunity,
    overall_accuracy_equality,
    predictive_equality,
    predictive_parity,
    statistical_parity,
    treatment_equality,
    total_fairness,
    balance_for_positive_class,
    equalized_odds,
    balance_for_negative_class,
    test_fairness,
    well_calibration,
)


def disparate_impact_remover(
    data_path: str,
    target_column: str,
    sensitive_column: str,
    repair_level: float = 1.0,
    test_split_percent: float = 20.0,
    model_params: dict = None,
    privileged_classes=None,
    unprivileged_classes=None,
    random_state: int = 42,
):
    """
    Apply Disparate Impact Remover and train/evaluate a model on the original and repaired datasets.

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
    - privileged_classes: list
        List of privileged classes for the sensitive attribute.
    - unprivileged_classes: list
        List of unprivileged classes for the sensitive attribute.
    - random_state: int
        Random seed for reproducibility.

    Returns:
    - metrics: dict
        Dictionary containing model performance metrics before and after repair.
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
    split_index = int(len(data) * ((1 - test_split_percent)))
    train_data = data.iloc[:split_index]
    test_data = data.iloc[split_index:]

    X_train, y_train = (
        train_data.drop(columns=[target_column]),
        train_data[target_column],
    )
    X_test, y_test = test_data.drop(columns=[target_column]), test_data[target_column]
    target_encoder = LabelEncoder()
    y_train = target_encoder.fit_transform(y_train)
    y_test = target_encoder.fit_transform(y_test)

    with open("target_label_encoder.pkl", "wb") as f:
        pickle.dump(target_encoder, f)
    feature_encoders = {}
    categorical_cols = X_train.select_dtypes(include=["object"]).columns
    for col in categorical_cols:
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col])
        X_test[col] = le.transform(X_test[col])
        feature_encoders[col] = le
    with open("label_encoders.pkl", "wb") as f:
        pickle.dump(feature_encoders, f)

    train_ds = StandardDataset(
        df=pd.concat([X_train, pd.Series(y_train, name=target_column)], axis=1),
        label_name=target_column,
        favorable_classes=[1],
        protected_attribute_names=[sensitive_column],
        privileged_classes=privileged_classes,
        features_to_keep=X_train.columns.tolist(),
    )

    di_remover = DisparateImpactRemover(repair_level=repair_level)

    train_ds_repaired = di_remover.fit_transform(train_ds)

    X_train_repaired = train_ds_repaired.features
    y_train_repaired = train_ds_repaired.labels.ravel()

    model_original = XGBClassifier(random_state=random_state, **model_params)
    model_repaired = XGBClassifier(random_state=random_state, **model_params)

    model_original.fit(X_train, y_train)
    y_pred_original = model_original.predict(X_test)

    model_repaired.fit(X_train_repaired, y_train_repaired)
    y_pred_repaired = model_repaired.predict(X_test)
    probabilities = model_repaired.predict_proba(X_test)
    accuracy_original = accuracy_score(y_test, y_pred_original)
    accuracy_repaired = accuracy_score(y_test, y_pred_repaired)
    cm_original = confusion_matrix(y_test, y_pred_original)
    cm_repaired = confusion_matrix(y_test, y_pred_repaired)
    tn, fp, fn, tp = cm_repaired.ravel()

    sensitive_column_column = test_data[sensitive_column]

    y_test_with_sensitive = pd.DataFrame(
        {"y_test": y_test, f"{sensitive_column}": sensitive_column_column}
    )

    predictions_with_sensitive = pd.DataFrame(
        {"predictions": y_pred_repaired, f"{sensitive_column}": sensitive_column_column}
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

    precision_repaired = precision_score(y_test, y_pred_repaired)
    recall_repaired = recall_score(y_test, y_pred_repaired)
    f1_score_repaired = f1_score(y_test, y_pred_repaired)

    altered_features = np.sum((X_train.values != X_train_repaired).sum(axis=1))
    total_features = X_train.size
    alteration_ratio = float(altered_features) / float(total_features)

    metrics = {
        "predictions": y_pred_repaired.tolist(),
        "probabilities": probabilities.tolist(),
        "accuracy": float(accuracy_repaired),
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
