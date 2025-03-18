import os
import pickle
import pandas as pd
import tempfile
import sys
import subprocess

python_executable = sys.executable
import time
from aif360.algorithms import Transformer

import numpy as np


class PrejudiceRemover(Transformer):
    """Prejudice remover is an in-processing technique that adds a
    discrimination-aware regularization term to the learning objective [6]_.

    References:
        .. [6] T. Kamishima, S. Akaho, H. Asoh, and J. Sakuma, "Fairness-Aware
           Classifier with Prejudice Remover Regularizer," Joint European
           Conference on Machine Learning and Knowledge Discovery in Databases,
           2012.

    """

    def __init__(self, eta=1.0, sensitive_attr="", class_attr=""):
        """
        Args:
            eta (double, optional): fairness penalty parameter
            sensitive_attr (str, optional): name of protected attribute
            class_attr (str, optional): label name
        """
        super(PrejudiceRemover, self).__init__(
            eta=eta, sensitive_attr=sensitive_attr, class_attr=class_attr
        )
        self.eta = eta
        self.sensitive_attr = sensitive_attr
        self.class_attr = class_attr

    def _create_file_in_kamishima_format(
        self,
        df,
        class_attr,
        positive_class_val,
        sensitive_attrs,
        single_sensitive,
        privileged_vals,
    ):
        """Format the data for the Kamishima code and save it."""
        x = []
        for col in df:
            if col != class_attr and col not in sensitive_attrs:
                x.append(np.array(df[col].values, dtype=np.float64))
        x.append(np.array(single_sensitive.isin(privileged_vals), dtype=np.float64))
        x.append(np.array(df[class_attr] == positive_class_val, dtype=np.float64))

        fd, name = tempfile.mkstemp()
        os.close(fd)
        np.savetxt(name, np.array(x).T)
        return name

    def fit(self, dataset):
        """Learns the regularized logistic regression model.

        Args:
            dataset (BinaryLabelDataset): Dataset containing true labels.

        Returns:
            PrejudiceRemover: Returns self.
        """
        data = np.column_stack([dataset.features, dataset.labels])
        columns = dataset.feature_names + dataset.label_names
        train_df = pd.DataFrame(data=data, columns=columns)

        all_sensitive_attributes = dataset.protected_attribute_names

        if not self.sensitive_attr:
            self.sensitive_attr = all_sensitive_attributes[0]
        self.sensitive_ind = all_sensitive_attributes.index(self.sensitive_attr)

        sens_df = pd.Series(
            dataset.protected_attributes[:, self.sensitive_ind],
            name=self.sensitive_attr,
        )

        if not self.class_attr:
            self.class_attr = dataset.label_names[0]

        fd, model_name = tempfile.mkstemp()
        os.close(fd)
        train_name = self._create_file_in_kamishima_format(
            train_df,
            self.class_attr,
            dataset.favorable_label,
            all_sensitive_attributes,
            sens_df,
            dataset.privileged_protected_attributes[self.sensitive_ind],
        )
        k_path = os.path.dirname(os.path.abspath(__file__))
        train_pr = os.path.join(k_path, "kamfadm-2012ecmlpkdd", "train_pr.py")
        subprocess.call(
            [
                "python3",
                train_pr,
                "-e",
                str(self.eta),
                "-i",
                train_name,
                "-o",
                model_name,
                "--quiet",
            ]
        )
        os.unlink(train_name)

        self.model_name = model_name

        return self

    def predict(self, dataset):
        """Obtain the predictions for the provided dataset using the learned
        prejudice remover model.

        Args:
            dataset (BinaryLabelDataset): Dataset containing labels that needs
                to be transformed.
        Returns:
            dataset (BinaryLabelDataset): Transformed dataset.
        """
        data = np.column_stack([dataset.features, dataset.labels])
        columns = dataset.feature_names + dataset.label_names
        test_df = pd.DataFrame(data=data, columns=columns)
        sens_df = pd.Series(
            dataset.protected_attributes[:, self.sensitive_ind],
            name=self.sensitive_attr,
        )

        fd, output_name = tempfile.mkstemp()
        os.close(fd)

        test_name = self._create_file_in_kamishima_format(
            test_df,
            self.class_attr,
            dataset.favorable_label,
            dataset.protected_attribute_names,
            sens_df,
            dataset.privileged_protected_attributes[self.sensitive_ind],
        )

        # ADDED FOLLOWING LINE to get absolute path of this file, i.e.
        # prejudice_remover.py
        k_path = os.path.dirname(os.path.abspath(__file__))
        predict_lr = os.path.join(k_path, "kamfadm-2012ecmlpkdd", "predict_lr.py")
        # changed paths in the calls below to (a) specify path of train_pr,
        # predict_lr RELATIVE to this file, and (b) compute & use absolute path,
        # and (c) replace python3 with python
        subprocess.call(
            [
                "python3",
                predict_lr,
                "-i",
                test_name,
                "-m",
                self.model_name,
                "-o",
                output_name,
                "--quiet",
            ]
        )
        os.unlink(test_name)
        m = np.loadtxt(output_name)
        os.unlink(output_name)

        pred_dataset = dataset.copy()
        # Columns of Outputs: (as per Kamishima implementation predict_lr.py)
        # 0. true sample class number
        # 1. predicted class number
        # 2. sensitive feature
        # 3. class 0 probability
        # 4. class 1 probability
        pred_dataset.labels = m[:, [1]]
        pred_dataset.scores = m[:, [4]]

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


def apply_prejudice_remover(
    data_path: str,
    target_column: str,
    sensitive_column: str,
    eta: float = 0.1,
    test_split_percent: float = 20.0,
    model_params: dict = None,
    privileged_classes=None,
    unprivileged_classes=None,
    random_state: int = 42,
):
    """
    Apply Prejudice Remover and train/evaluate a model on the original and repaired datasets.

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
    model = PrejudiceRemover(eta=eta, sensitive_attr=sensitive_column)
    start_time = time.time()

    model.fit(train_ds)
    end_time = time.time()
    total_time = end_time - start_time

    with open("pjr_model.pkl", "wb") as f:
        pickle.dump(model, f)

    model_size = os.path.getsize("pjr_model.pkl")

    test_preds = model.predict(test_ds)
    test_preds = test_preds.labels.flatten()
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
