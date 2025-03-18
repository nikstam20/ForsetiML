import os
import pickle
import pandas as pd
import tempfile
import sys
import subprocess

python_executable = sys.executable
import time
import numpy as np

from aif360.datasets import BinaryLabelDataset
from aif360.algorithms import Transformer


class ARTClassifier(Transformer):
    """Wraps an instance of an :obj:`art.classifiers.Classifier` to extend
    :obj:`~aif360.algorithms.Transformer`.
    """

    def __init__(self, art_classifier):
        """Initialize ARTClassifier.

        Args:
            art_classifier (art.classifier.Classifier): A Classifier
                object from the `adversarial-robustness-toolbox`_.

        .. _adversarial-robustness-toolbox:
           https://github.com/Trusted-AI/adversarial-robustness-toolbox
        """
        super(ARTClassifier, self).__init__(art_classifier=art_classifier)
        self._art_classifier = art_classifier

    def fit(self, dataset, batch_size=128, nb_epochs=20):
        """Train a classifer on the input.

        Args:
            dataset (Dataset): Training dataset.
            batch_size (int): Size of batches (passed through to ART).
            nb_epochs (int): Number of epochs to use for training (passed
                through to ART).

        Returns:
            ARTClassifier: Returns self.
        """
        self._art_classifier.fit(
            dataset.features, dataset.labels, batch_size=batch_size, nb_epochs=nb_epochs
        )
        return self

    def predict(self, dataset, logits=False):
        """Perform prediction for the input.

        Args:
            dataset (Dataset): Test dataset.
            logits (bool, optional): True is prediction should be done at the
                logits layer (passed through to ART).

        Returns:
            Dataset: Dataset with predicted labels in the `labels` field.
        """
        pred_labels = self._art_classifier.predict(
            dataset.features, dataset.labels, logits=logits
        )

        if isinstance(dataset, BinaryLabelDataset):
            pred_labels = np.argmax(pred_labels, axis=1).reshape((-1, 1))

        pred_dataset = dataset.copy()
        pred_dataset.labels = pred_labels

        return pred_dataset


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


def apply_art(
    data_path: str,
    target_column: str,
    sensitive_column: str,
    test_split_percent: float = 20.0,
    model_params: dict = None,
    privileged_classes=None,
    unprivileged_classes=None,
    random_state: int = 42,
):
    """
    Apply ART and train/evaluate a model on the original and repaired datasets.

    Parameters:
    - data_path: str
        Path to the input CSV dataset.
    - target_column: str
        Name of the target column in the dataset.
    - sensitive_column: str
        Name of the sensitive attribute to debias.
    - eta: float
        Fairness regularization parameter (higher values = more fairness).
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
    if privileged_classes is None:
        raise ValueError(
            "privileged_classes and unprivileged_classes must be provided."
        )

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")
    data = pd.read_csv(data_path)
    data.loc[data[sensitive_column] == privileged_classes, sensitive_column] = 1
    data.loc[data[sensitive_column] == unprivileged_classes, sensitive_column] = 0
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
    y_train_encoded = target_encoder.fit_transform(y_train)
    y_test_encoded = target_encoder.fit_transform(y_test)

    with open("target_label_encoder.pkl", "wb") as f:
        pickle.dump(target_encoder, f)
    feature_encoders = {}
    categorical_cols = X_train.select_dtypes(include=["object"]).columns
    for col in categorical_cols:
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col])
        X_test[col] = le.transform(X_test[col])
        feature_encoders[col] = le
    privileged_classes = [[1]]

    def z_score_normalize(df, feature_names):
        for feature in feature_names:
            mean = df[feature].mean()
            std = df[feature].std()
            df[feature] = (df[feature] - mean) / std
        return df

    X_train = z_score_normalize(X_train, X_train.columns)
    X_test = z_score_normalize(X_test, X_test.columns)
    train_ds = StandardDataset(
        df=pd.concat([X_train, pd.Series(y_train_encoded, name=target_column)], axis=1),
        label_name=target_column,
        favorable_classes=[1.0],
        protected_attribute_names=[sensitive_column],
        privileged_classes=privileged_classes,
        categorical_features=X_train.select_dtypes(include=["object"]).columns,
        features_to_keep=X_train.columns.tolist(),
    )

    test_ds = StandardDataset(
        df=pd.concat([X_test, pd.Series(y_test, name=target_column)], axis=1),
        label_name=target_column,
        favorable_classes=[1.0],
        protected_attribute_names=[sensitive_column],
        privileged_classes=privileged_classes,
        categorical_features=X_test.select_dtypes(include=["object"]).columns,
        features_to_keep=X_test.columns.tolist(),
    )
    model = ARTClassifier(sensitive_attr=sensitive_column)
    start_time = time.time()

    model.fit(train_ds)
    end_time = time.time()
    total_time = end_time - start_time

    with open("pjr_model.pkl", "wb") as f:
        pickle.dump(model, f)

    model_size = os.path.getsize("pjr_model.pkl")

    test_preds = model.predict(test_ds)
    test_preds = test_preds.labels.flatten()
    print(test_preds)
    accuracy_repaired = accuracy_score(y_test_encoded, test_preds)
    cm_repaired = confusion_matrix(y_test_encoded, test_preds)
    tn, fp, fn, tp = cm_repaired.ravel()

    sensitive_column_column = test_data[sensitive_column]

    y_test_with_sensitive = pd.DataFrame(
        {"y_test": y_test_encoded, f"{sensitive_column}": sensitive_column_column}
    )

    predictions_with_sensitive = pd.DataFrame(
        {"predictions": test_preds, f"{sensitive_column}": sensitive_column_column}
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

    precision_repaired = precision_score(y_test_encoded, test_preds)
    recall_repaired = recall_score(y_test_encoded, test_preds)
    f1_score_repaired = f1_score(y_test_encoded, test_preds)

    metrics = {
        "predictions": test_preds.tolist(),
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
        "training_time": total_time,
        "model_size": model_size,
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
    }

    return metrics
