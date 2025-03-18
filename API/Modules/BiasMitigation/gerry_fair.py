import os
import pickle
import pandas as pd
import tempfile
import sys
import subprocess

python_executable = sys.executable
import time

# Copyright 2019 Seth V. Neel, Michael J. Kearns, Aaron L. Roth, Zhiwei Steven Wu
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.
"""Class GerryFairClassifier implementing the 'FairFictPlay' Algorithm of [KRNW18].

This module contains functionality to instantiate, fit, and predict
using the FairFictPlay algorithm of:
https://arxiv.org/abs/1711.05144
It also contains the ability to audit arbitrary classifiers for
rich subgroup unfairness, where rich subgroups are defined by hyperplanes
over the sensitive attributes. This iteration of the codebase supports hyperplanes, trees,
kernel methods, and support vector machines. For usage examples refer to examples/gerry_plots.ipynb
"""


import copy
from aif360.algorithms.inprocessing.gerryfair import heatmap
from aif360.algorithms.inprocessing.gerryfair.clean import array_to_tuple
from aif360.algorithms.inprocessing.gerryfair.learner import Learner
from aif360.algorithms.inprocessing.gerryfair.auditor import *
from aif360.algorithms.inprocessing.gerryfair.classifier_history import (
    ClassifierHistory,
)
from aif360.algorithms import Transformer


class GerryFairClassifier(Transformer):
    """Model is an algorithm for learning classifiers that are fair with respect
    to rich subgroups.

    Rich subgroups are defined by (linear) functions over the sensitive
    attributes, and fairness notions are statistical: false positive, false
    negative, and statistical parity rates. This implementation uses a max of
    two regressions as a cost-sensitive classification oracle, and supports
    linear regression, support vector machines, decision trees, and kernel
    regression. For details see:

    References:
        .. [1] "Preventing Fairness Gerrymandering: Auditing and Learning for
           Subgroup Fairness." Michale Kearns, Seth Neel, Aaron Roth, Steven Wu.
           ICML '18.
        .. [2] "An Empirical Study of Rich Subgroup Fairness for Machine
           Learning". Michael Kearns, Seth Neel, Aaron Roth, Steven Wu. FAT '19.
    """

    def __init__(
        self,
        C=10,
        printflag=False,
        heatmapflag=False,
        heatmap_iter=10,
        heatmap_path=".",
        max_iters=10,
        gamma=0.01,
        fairness_def="FP",
        predictor=linear_model.LinearRegression(),
    ):
        """Initialize Model Object and set hyperparameters.

        Args:
            C: Maximum L1 Norm for the Dual Variables (hyperparameter)
            printflag: Print Output Flag
            heatmapflag: Save Heatmaps every heatmap_iter Flag
            heatmap_iter: Save Heatmaps every heatmap_iter
            heatmap_path: Save Heatmaps path
            max_iters: Time Horizon for the fictitious play dynamic.
            gamma: Fairness Approximation Paramater
            fairness_def: Fairness notion, FP, FN, SP.
            errors: see fit()
            fairness_violations: see fit()
            predictor: Hypothesis class for the Learner. Supports LR, SVM, KR,
                Trees.
        """

        super(GerryFairClassifier, self).__init__()
        self.C = C
        self.printflag = printflag
        self.heatmapflag = heatmapflag
        self.heatmap_iter = heatmap_iter
        self.heatmap_path = heatmap_path
        self.max_iters = max_iters
        self.gamma = gamma
        self.fairness_def = fairness_def
        self.predictor = predictor
        self.classifiers = None
        self.errors = None
        self.fairness_violations = None
        if self.fairness_def not in ["FP", "FN"]:
            raise Exception(
                "This metric is not yet supported for learning. Metric specified: {}.".format(
                    self.fairness_def
                )
            )

    def fit(self, dataset, early_termination=True):
        """Run Fictitious play to compute the approximately fair classifier.

        Args:
            dataset: dataset object with its own class definition in datasets
                folder inherits from class StandardDataset.
            early_termination: Terminate Early if Auditor can't find fairness
                violation of more than gamma.
        Returns:
            Self
        """

        # defining variables and data structures for algorithm
        X, X_prime, y = clean.extract_df_from_ds(dataset)
        learner = Learner(X, y, self.predictor)
        auditor = Auditor(dataset, self.fairness_def)
        history = ClassifierHistory()

        # initialize variables
        n = X.shape[0]
        costs_0, costs_1, X_0 = auditor.initialize_costs(n)
        metric_baseline = 0
        predictions = [0.0] * n

        # scaling variables for heatmap
        vmin = None
        vmax = None

        # print output variables
        errors = []
        fairness_violations = []

        iteration = 1
        while iteration < self.max_iters:
            # learner's best response: solve the CSC problem, get mixture decisions on X to update prediction probabilities
            history.append_classifier(learner.best_response(costs_0, costs_1))
            error, predictions = learner.generate_predictions(
                history.get_most_recent(), predictions, iteration
            )
            # auditor's best response: find group, update costs
            metric_baseline = auditor.get_baseline(y, predictions)
            group = auditor.get_group(predictions, metric_baseline)
            costs_0, costs_1 = auditor.update_costs(
                costs_0, costs_1, group, self.C, iteration, self.gamma
            )

            # outputs
            errors.append(error)
            fairness_violations.append(group.weighted_disparity)
            self.print_outputs(iteration, error, group)
            vmin, vmax = self.save_heatmap(
                iteration, dataset, history.get_most_recent().predict(X), vmin, vmax
            )
            iteration += 1

            # early termination:
            if (
                early_termination
                and (len(errors) >= 5)
                and (
                    (errors[-1] == errors[-2])
                    or fairness_violations[-1] == fairness_violations[-2]
                )
                and fairness_violations[-1] < self.gamma
            ):
                iteration = self.max_iters

        self.classifiers = history.classifiers
        self.errors = errors
        self.fairness_violations = fairness_violations
        return self

    def predict(self, dataset, threshold=0.5):
        """Return dataset object where labels are the predictions returned by
        the fitted model.

        Args:
            dataset: dataset object with its own class definition in datasets
                folder inherits from class StandardDataset.
            threshold: The positive prediction cutoff for the soft-classifier.

        Returns:
            dataset_new: modified dataset object where the labels attribute are
            the predictions returned by the self model
        """

        # Generates predictions.
        dataset_new = copy.deepcopy(dataset)
        data, _, _ = clean.extract_df_from_ds(dataset_new)
        num_classifiers = len(self.classifiers)
        y_hat = None
        for hyp in self.classifiers:
            new_predictions = hyp.predict(data) / num_classifiers
            if y_hat is None:
                y_hat = new_predictions
            else:
                y_hat = np.add(y_hat, new_predictions)
        if threshold:
            dataset_new.labels = np.asarray([1 if y >= threshold else 0 for y in y_hat])
        else:
            dataset_new.labels = np.asarray([y for y in y_hat])
        dataset_new.labels.resize(dataset.labels.shape, refcheck=True)
        return dataset_new

    def print_outputs(self, iteration, error, group):
        """Helper function to print outputs at each iteration of fit.

        Args:
            iteration: current iter
            error: most recent error
            group: most recent group found by the auditor
        """

        if self.printflag:
            print(
                "iteration: {}, error: {}, fairness violation: {}, violated group size: {}".format(
                    int(iteration), error, group.weighted_disparity, group.group_size
                )
            )

    def save_heatmap(self, iteration, dataset, predictions, vmin, vmax):
        """Helper Function to save the heatmap.

        Args:
            iteration: current iteration
            dataset: dataset object with its own class definition in datasets
                folder inherits from class StandardDataset.
            predictions: predictions of the model self on dataset.
            vmin: see documentation of heatmap.py heat_map function
            vmax: see documentation of heatmap.py heat_map function

        Returns:
            (vmin, vmax)
        """

        X, X_prime, y = clean.extract_df_from_ds(dataset)
        # save heatmap every heatmap_iter iterations or the last iteration
        if self.heatmapflag and (iteration % self.heatmap_iter) == 0:
            # initial heat map
            X_prime_heat = X_prime.iloc[:, 0:2]
            eta = 0.1
            minmax = heatmap.heat_map(
                X,
                X_prime_heat,
                y,
                predictions,
                eta,
                self.heatmap_path + "/heatmap_iteration_{}".format(iteration),
                vmin,
                vmax,
            )
            if iteration == 1:
                vmin = minmax[0]
                vmax = minmax[1]
        return vmin, vmax

    def generate_heatmap(
        self, dataset, predictions, vmin=None, vmax=None, cols_index=[0, 1], eta=0.1
    ):
        """Helper Function to generate the heatmap at the current time.

        Args:
            iteration:current iteration
            dataset: dataset object with its own class definition in datasets
                folder inherits from class StandardDataset.
            predictions: predictions of the model self on dataset.
            vmin: see documentation of heatmap.py heat_map function
            vmax: see documentation of heatmap.py heat_map function
        """

        X, X_prime, y = clean.extract_df_from_ds(dataset)
        # save heatmap every heatmap_iter iterations or the last iteration
        X_prime_heat = X_prime.iloc[:, cols_index]
        minmax = heatmap.heat_map(
            X, X_prime_heat, y, predictions, eta, self.heatmap_path, vmin, vmax
        )

    def pareto(self, dataset, gamma_list):
        """Assumes Model has FP specified for metric. Trains for each value of
        gamma, returns error, FP (via training), and FN (via auditing) values.

        Args:
            dataset: dataset object with its own class definition in datasets
                folder inherits from class StandardDataset.
            gamma_list: the list of gamma values to generate the pareto curve

        Returns:
            list of errors, list of fp violations of those models, list of fn
            violations of those models
        """

        C = self.C
        max_iters = self.max_iters

        # Store errors and fp over time for each gamma

        # change var names, but no real dependence on FP logic
        all_errors = []
        all_fp_violations = []
        all_fn_violations = []
        self.C = C
        self.max_iters = max_iters

        auditor = Auditor(dataset, "FN")
        for g in gamma_list:
            self.gamma = g
            fitted_model = self.fit(dataset, early_termination=True)
            errors, fairness_violations = (
                fitted_model.errors,
                fitted_model.fairness_violations,
            )
            predictions = array_to_tuple((self.predict(dataset)).labels)
            _, fn_violation = auditor.audit(predictions)
            all_errors.append(errors[-1])
            all_fp_violations.append(fairness_violations[-1])
            all_fn_violations.append(fn_violation)

        return all_errors, all_fp_violations, all_fn_violations


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


def apply_gerry_fair(
    data_path: str,
    target_column: str,
    sensitive_column: str,
    C: int = 10,
    max_iters: int = 10,
    gamma: float = 0.01,
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
    model = GerryFairClassifier(C=10, max_iters=10, gamma=0.01)
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
