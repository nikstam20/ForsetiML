import os
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.preprocessing import LabelEncoder
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


def deterministic_reranking(
    data_path: str,
    target_column: str,
    sensitive_column: str,
    privileged_classes,
    unprivileged_classes,
    test_split_percent: float = 20.0,
    random_state: int = 42,
):
    """
    Apply Deterministic Reranking postprocessing using a pre-trained model and evaluate
    performance on the original and adjusted datasets.

    Parameters:
    - data_path: str - Path to the input CSV dataset.
    - target_column: str - Name of the target column in the dataset.
    - sensitive_column: str - Name of the sensitive attribute to debias.
    - privileged_classes: list - List of privileged classes for the sensitive attribute.
    - unprivileged_classes: list - List of unprivileged classes for the sensitive attribute.
    - test_split_percent: float - Proportion of data to be used as test set (percentage).
    - random_state: int - Random seed for reproducibility.

    Returns:
    - metrics: dict - Dictionary containing model performance metrics before and after reranking.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")
    data = pd.read_csv(data_path)

    if target_column not in data.columns or sensitive_column not in data.columns:
        raise ValueError(
            f"Target column '{target_column}' or sensitive column '{sensitive_column}' not found in data."
        )

    if not os.path.exists("pretrained_model.pkl"):
        raise FileNotFoundError("Pre-trained model file not found.")
    with open("pretrained_model.pkl", "rb") as f:
        model = pickle.load(f)

    data[sensitive_column] = data[sensitive_column].apply(
        lambda x: 1 if x in privileged_classes else 0
    )

    label_encoder = LabelEncoder()
    data[target_column] = label_encoder.fit_transform(data[target_column])

    split_index = int(len(data) * (1 - test_split_percent / 100))
    test_data = data.iloc[split_index:]

    X_test = test_data.drop(columns=[target_column])
    y_test = test_data[target_column]

    if not os.path.exists("feature_encoders.pkl"):
        raise FileNotFoundError("Feature encoders file not found.")
    with open("feature_encoders.pkl", "rb") as f:
        feature_encoders = pickle.load(f)

    categorical_cols = X_test.select_dtypes(include=["object"]).columns
    for col in categorical_cols:
        if col in feature_encoders:
            le = feature_encoders[col]
            X_test[col] = le.transform(X_test[col])
        else:
            raise ValueError(
                f"Encoder for column '{col}' is missing in feature_encoders.pkl."
            )

    initial_predictions = model.predict(X_test)

    results_df = pd.DataFrame(
        {
            "true_label": y_test,
            "prediction": initial_predictions,
            "sensitive_attribute": X_test[sensitive_column],
        }
    )

    privileged_df = results_df[results_df["sensitive_attribute"] == 1]
    unprivileged_df = results_df[results_df["sensitive_attribute"] == 0]

    privileged_df = privileged_df.sort_values(by="prediction", ascending=False)
    unprivileged_df = unprivileged_df.sort_values(by="prediction", ascending=False)

    total_positives = results_df["prediction"].sum()
    privileged_positive_quota = int(
        total_positives * len(privileged_df) / len(results_df)
    )
    unprivileged_positive_quota = int(
        total_positives * len(unprivileged_df) / len(results_df)
    )

    privileged_df["adjusted_prediction"] = [1] * privileged_positive_quota + [0] * (
        len(privileged_df) - privileged_positive_quota
    )
    unprivileged_df["adjusted_prediction"] = [1] * unprivileged_positive_quota + [0] * (
        len(unprivileged_df) - unprivileged_positive_quota
    )

    adjusted_results_df = pd.concat([privileged_df, unprivileged_df]).sort_index()

    adjusted_predictions = adjusted_results_df["adjusted_prediction"].values

    accuracy_original = accuracy_score(y_test, initial_predictions)
    accuracy_adjusted = accuracy_score(y_test, adjusted_predictions)

    precision_original = precision_score(y_test, initial_predictions)
    precision_adjusted = precision_score(y_test, adjusted_predictions)

    recall_original = recall_score(y_test, initial_predictions)
    recall_adjusted = recall_score(y_test, adjusted_predictions)

    f1_original = f1_score(y_test, initial_predictions)
    f1_adjusted = f1_score(y_test, adjusted_predictions)

    cm_original = confusion_matrix(y_test, initial_predictions)
    cm_adjusted = confusion_matrix(y_test, adjusted_predictions)

    conditional_stat_parity = conditional_statistical_parity(
        y_test, adjusted_predictions, X_test[sensitive_column]
    )
    conditional_accuracy_eq = conditional_use_accuracy_equality(
        y_test, adjusted_predictions, X_test[sensitive_column]
    )
    equal_neg_predictive_val = equal_negative_predictive_value(
        y_test, adjusted_predictions, X_test[sensitive_column]
    )
    equal_opportunity_metric = equal_opportunity(
        y_test, adjusted_predictions, X_test[sensitive_column]
    )
    overall_accuracy_eq = overall_accuracy_equality(
        y_test, adjusted_predictions, X_test[sensitive_column]
    )
    predictive_eq = predictive_equality(
        y_test, adjusted_predictions, X_test[sensitive_column]
    )
    statistical_parity_metric = statistical_parity(
        y_test, adjusted_predictions, X_test[sensitive_column]
    )
    treatment_eq = treatment_equality(
        y_test, adjusted_predictions, X_test[sensitive_column]
    )
    balance_positive_class = balance_for_positive_class(
        y_test, adjusted_predictions, X_test[sensitive_column]
    )
    balance_negative_class = balance_for_negative_class(
        y_test, adjusted_predictions, X_test[sensitive_column]
    )
    equalized_odds_metric = equalized_odds(
        y_test, adjusted_predictions, X_test[sensitive_column]
    )

    altered_features = np.sum(initial_predictions != adjusted_predictions)
    total_features = len(initial_predictions)
    alteration_ratio = altered_features / total_features

    metrics = {
        "accuracy_original": float(accuracy_original),
        "accuracy_adjusted": float(accuracy_adjusted),
        "precision_original": float(precision_original),
        "precision_adjusted": float(precision_adjusted),
        "recall_original": float(recall_original),
        "recall_adjusted": float(recall_adjusted),
        "f1_original": float(f1_original),
        "f1_adjusted": float(f1_adjusted),
        "confusion_matrix_original": {
            "true_negatives": int(cm_original[0, 0]),
            "false_positives": int(cm_original[0, 1]),
            "false_negatives": int(cm_original[1, 0]),
            "true_positives": int(cm_original[1, 1]),
        },
        "confusion_matrix_adjusted": {
            "true_negatives": int(cm_adjusted[0, 0]),
            "false_positives": int(cm_adjusted[0, 1]),
            "false_negatives": int(cm_adjusted[1, 0]),
            "true_positives": int(cm_adjusted[1, 1]),
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
        "balance_for_negative_class": balance_negative_class,
        "equalized_odds": equalized_odds_metric,
        "alteration_ratio": alteration_ratio,
    }

    return metrics
