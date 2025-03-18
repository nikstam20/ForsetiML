import os
import pandas as pd
import numpy as np
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


def flip_labels(
    data_path: str,
    target_column: str,
    sensitive_column: str,
    flip_rate: float = 0.1,
    random_state: int = 42,
):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")
    data = pd.read_csv(data_path)

    for value in data[sensitive_column].unique():
        group_indices = data[data[sensitive_column] == value].index
        flip_count = int(len(group_indices) * flip_rate)
        flip_indices = np.random.choice(group_indices, size=flip_count, replace=False)
        data.loc[flip_indices, target_column] = (
            1 - data.loc[flip_indices, target_column]
        )

    X = data.drop(columns=[target_column])
    y = data[target_column]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )
    y_train_encoded = LabelEncoder().fit_transform(y_train)
    y_test_encoded = LabelEncoder().transform(y_test)

    model = XGBClassifier(random_state=random_state)
    model.fit(X_train, y_train_encoded)
    y_pred = model.predict(X_test)
    probabilities = model.predict_proba(X_test)

    accuracy = accuracy_score(y_test_encoded, y_pred)
    cm = confusion_matrix(y_test_encoded, y_pred)
    tn, fp, fn, tp = cm.ravel()
    precision = precision_score(y_test_encoded, y_pred)
    recall = recall_score(y_test_encoded, y_pred)
    f1 = f1_score(y_test_encoded, y_pred)

    metrics = {
        "predictions": y_pred.tolist(),
        "probabilities": probabilities.tolist(),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        },
        "conditional_statistical_parity": conditional_statistical_parity(
            y_test, y_pred, sensitive_column
        ),
        "conditional_use_accuracy_equality": conditional_use_accuracy_equality(
            y_test, y_pred, sensitive_column
        ),
        "equal_negative_predictive_value": equal_negative_predictive_value(
            y_test, y_pred, sensitive_column
        ),
        "equal_opportunity": equal_opportunity(y_test, y_pred, sensitive_column),
        "overall_accuracy_equality": overall_accuracy_equality(
            y_test, y_pred, sensitive_column
        ),
        "predictive_equality": predictive_equality(y_test, y_pred, sensitive_column),
        "statistical_parity": statistical_parity(y_test, y_pred, sensitive_column),
        "treatment_equality": treatment_equality(y_test, y_pred, sensitive_column),
        "balance_for_positive_class": balance_for_positive_class(
            y_test, y_pred, sensitive_column
        ),
        "equalized_odds": equalized_odds(y_test, y_pred, sensitive_column),
        "balance_for_negative_class": balance_for_negative_class(
            y_test, y_pred, sensitive_column
        ),
        "test_fairness": test_fairness(y_test, y_pred, sensitive_column),
        "well_calibration": well_calibration(y_test, y_pred, sensitive_column),
        "alteration_ratio": float(flip_count) / len(group_indices),
    }

    return metrics
