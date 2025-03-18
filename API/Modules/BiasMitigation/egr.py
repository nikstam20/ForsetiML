import os
import pickle
import pandas as pd
import tempfile
import sys
import subprocess

python_executable = sys.executable
import time

"""
The code for ExponentiatedGradientReduction wraps the source class
fairlearn.reductions.ExponentiatedGradient
available in the https://github.com/fairlearn/fairlearn library
licensed under the MIT Licencse, Copyright Microsoft Corporation
"""
from logging import warning

import pandas as pd

from aif360.algorithms import Transformer
from aif360.sklearn.inprocessing import ExponentiatedGradientReduction as skExpGradRed
from fairlearn.reductions import DemographicParity


class ExponentiatedGradientReduction(Transformer):
    """Exponentiated gradient reduction for fair classification.

    Exponentiated gradient reduction is an in-processing technique that reduces
    fair classification to a sequence of cost-sensitive classification problems,
    returning a randomized classifier with the lowest empirical error subject to
    fair classification constraints [#agarwal18]_.

    References:
        .. [#agarwal18] `A. Agarwal, A. Beygelzimer, M. Dudik, J. Langford, and
           H. Wallach, "A Reductions Approach to Fair Classification,"
           International Conference on Machine Learning, 2018.
           <https://arxiv.org/abs/1803.02453>`_
    """

    def __init__(
        self,
        estimator,
        constraints,
        eps=0.01,
        max_iter=50,
        nu=None,
        eta0=2.0,
        run_linprog_step=True,
        drop_prot_attr=True,
    ):
        """
        Args:
            estimator: An estimator implementing methods
                ``fit(X, y, sample_weight)`` and ``predict(X)``, where ``X`` is
                the matrix of features, ``y`` is the vector of labels, and
                ``sample_weight`` is a vector of weights; labels ``y`` and
                predictions returned by ``predict(X)`` are either 0 or 1 -- e.g.
                scikit-learn classifiers.
            constraints (str or fairlearn.reductions.Moment): If string, keyword
                denoting the :class:`fairlearn.reductions.Moment` object
                defining the disparity constraints -- e.g., "DemographicParity"
                or "EqualizedOdds". For a full list of possible options see
                `self.model.moments`. Otherwise, provide the desired
                :class:`~fairlearn.reductions.Moment` object defining the
                disparity constraints.
            eps: Allowed fairness constraint violation; the solution is
                guaranteed to have the error within ``2*best_gap`` of the best
                error under constraint eps; the constraint violation is at most
                ``2*(eps+best_gap)``.
            T: Maximum number of iterations.
            nu: Convergence threshold for the duality gap, corresponding to a
                conservative automatic setting based on the statistical
                uncertainty in measuring classification error.
            eta_mul: Initial setting of the learning rate.
            run_linprog_step: If True each step of exponentiated gradient is
                followed by the saddle point optimization over the convex hull
                of classifiers returned so far.
            drop_prot_attr: Boolean flag indicating whether to drop protected
                attributes from training data.

        """
        super(ExponentiatedGradientReduction, self).__init__()

        # init model, set prot_attr during fit
        prot_attr = []
        self.model = skExpGradRed(
            prot_attr=prot_attr,
            estimator=estimator,
            constraints=constraints,
            eps=eps,
            max_iter=max_iter,
            nu=nu,
            eta0=eta0,
            run_linprog_step=run_linprog_step,
            drop_prot_attr=drop_prot_attr,
        )

    def fit(self, dataset):
        """Learns randomized model with less bias

        Args:
            dataset: (Binary label) Dataset containing true labels.

        Returns:
            ExponentiatedGradientReduction: Returns self.
        """
        # set prot_attr
        self.model.prot_attr = dataset.protected_attribute_names

        X_df = pd.DataFrame(dataset.features, columns=dataset.feature_names)
        Y = dataset.labels

        self.model.fit(X_df, Y)

        return self

    def predict(self, dataset):
        """Obtain the predictions for the provided dataset using the randomized
        model learned.

        Args:
            dataset: (Binary label) Dataset containing labels that needs to be
                transformed.

        Returns:
            dataset: Transformed (Binary label) dataset.
        """
        X_df = pd.DataFrame(dataset.features, columns=dataset.feature_names)

        dataset_new = dataset.copy()
        dataset_new.labels = self.model.predict(X_df).reshape(-1, 1)

        fav = int(dataset.favorable_label)
        try:
            # Probability of favorable label
            scores = self.model.predict_proba(X_df)[:, fav]
            dataset_new.scores = scores.reshape(-1, 1)
        except (AttributeError, NotImplementedError):
            warning(
                "dataset.scores not updated, underlying model does not "
                "support predict_proba"
            )

        return dataset_new


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
import xgboost as xgb
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


def apply_egr(
    data_path: str,
    target_column: str,
    sensitive_column: str,
    eps: float = 0.01,
    max_iter: int = 50,
    nu: str = None,
    eta0: float = 2.0,
    test_split_percent: float = 20.0,
    model_params: dict = None,
    privileged_classes=None,
    unprivileged_classes=None,
    random_state: int = 42,
):
    """
    Apply EGR and train/evaluate.

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

    xgb_estimator = xgb.XGBClassifier(
        use_label_encoder=False,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=random_state,
    )
    demographic_parity_constraint = DemographicParity()

    model = ExponentiatedGradientReduction(
        estimator=xgb_estimator,
        constraints=demographic_parity_constraint,
        eps=0.01,
        max_iter=50,
        nu=None,
        eta0=2.0,
    )
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
