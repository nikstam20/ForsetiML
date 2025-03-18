import os
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
from aif360.algorithms.preprocessing.optim_preproc import OptimPreproc
from aif360.algorithms.preprocessing.optim_preproc_helpers.opt_tools import OptTools
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


def optimized_preprocessing(
    data_path: str,
    target_column: str,
    sensitive_column: str,
    test_split_percent: float = 20.0,
    model_params: dict = None,
    privileged_classes=None,
    unprivileged_classes=None,
    random_state: int = 42,
    batch_size: int = 500,
):
    if model_params is None:
        model_params = {}
    if privileged_classes is None:
        raise ValueError(
            "privileged_classes and unprivileged_classes must be provided."
        )
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")
    data = pd.read_csv(data_path)
    data.loc[data[sensitive_column] == privileged_classes, sensitive_column] = 1
    data.loc[data[sensitive_column] == unprivileged_classes, sensitive_column] = 0
    for col in data.select_dtypes(include="number").columns:
        data[col] = data[col].fillna(data[col].median())

    for col in data.select_dtypes(include="object").columns:
        data[col] = data[col].fillna(data[col].mode())[0]
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

    privileged_classes = [[1]]

    train_ds = StandardDataset(
        df=pd.concat([X_train, pd.Series(y_train, name=target_column)], axis=1),
        label_name=target_column,
        favorable_classes=[1],
        protected_attribute_names=[sensitive_column],
        privileged_classes=privileged_classes,
        categorical_features=X_train.select_dtypes(include=["object"]).columns,
        features_to_keep=X_train.columns.tolist(),
    )

    def distortion_function(vold, vnew, sensitive_attrs, target_attr):
        """
        Generalized distortion function that computes the distortion based on changes in sensitive attributes
        and the target attribute, considering binary privileged (1) and unprivileged (0) classes, and positive
        (1) and negative (0) outcomes.

        Args:
            vold (dict): Dictionary containing old values {attr:value}.
            vnew (dict): Dictionary containing new values {attr:value}.
            sensitive_attrs (list): List of names of sensitive attributes.
            target_attr (str): Name of the target attribute.

        Returns:
            float: Distortion value based on the changes from old to new values.
        """
        distortion = 0.0

        sensitive_attr_change_cost = 1.0
        target_attr_change_cost = 2.0

        for attr in sensitive_attrs:
            if attr in vold and attr in vnew and vold[attr] != vnew[attr]:
                distortion += sensitive_attr_change_cost * (
                    1 if vold[attr] != vnew[attr] else 0
                )

        if target_attr in vold and target_attr in vnew:
            if vold[target_attr] != vnew[target_attr]:
                distortion += target_attr_change_cost * (
                    2 if vold[target_attr] > vnew[target_attr] else 1
                )

        return distortion

    from aif360.algorithms.preprocessing.optim_preproc_helpers.distortion_functions import (
        get_distortion_adult,
        get_distortion_german,
        get_distortion_compas,
    )

    privileged_groups = [{sensitive_column: 1}]
    unprivileged_groups = [{sensitive_column: 0}]
    optim_options = {
        "distortion_fun": get_distortion_adult,
        "epsilon": 0.05,
        "clist": [0.99, 1.99, 2.99],
        "dlist": [0.1, 0.05, 0],
    }
    OP = OptimPreproc(
        OptTools,
        optim_options,
        unprivileged_groups=unprivileged_groups,
        privileged_groups=privileged_groups,
    )

    OP = OP.fit(train_ds)
    dataset_transf_train = OP.transform(train_ds, transform_Y=True)
    dataset_transf_train = train_ds.align_datasets(dataset_transf_train)
    # train_ds_optimized = StandardDataset(
    #     df=pd.concat([pd.DataFrame(transformed_features, columns=train_ds.feature_names),
    #                   pd.Series(transformed_labels.ravel(), name=train_ds.label_names[0])], axis=1),
    #     label_name=train_ds.label_names[0],
    #     favorable_classes=[favorable_label],
    #     protected_attribute_names=train_ds.protected_attribute_names,
    #     privileged_classes=privileged_classes,
    #     features_to_keep=train_ds.feature_names
    # )
    train_ds_optimized = train_ds.copy()

    model_original = XGBClassifier(random_state=random_state, **model_params)
    model_optimized = XGBClassifier(random_state=random_state, **model_params)

    model_original.fit(X_train, y_train)
    y_pred_original = model_original.predict(X_test)

    model_optimized.fit(train_ds_optimized.features, train_ds_optimized.labels.ravel())
    y_pred_optimized = model_optimized.predict(X_test)
    probabilities = model_optimized.predict_proba(X_test)

    accuracy_optimized = accuracy_score(y_test, y_pred_optimized)
    cm_optimized = confusion_matrix(y_test, y_pred_optimized)
    tn, fp, fn, tp = cm_optimized.ravel()

    sensitive_column_column = test_data[sensitive_column]
    y_test_with_sensitive = pd.DataFrame(
        {"y_test": y_test, f"{sensitive_column}": sensitive_column_column}
    )

    predictions_with_sensitive = pd.DataFrame(
        {
            "predictions": y_pred_optimized,
            f"{sensitive_column}": sensitive_column_column,
        }
    )

    metrics = {
        "predictions": y_pred_optimized.tolist(),
        "probabilities": probabilities.tolist(),
        "accuracy": float(accuracy_optimized),
        "precision": precision_score(y_test, y_pred_optimized),
        "recall": recall_score(y_test, y_pred_optimized),
        "f1_score": f1_score(y_test, y_pred_optimized),
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        },
        "conditional_statistical_parity": conditional_statistical_parity(
            y_test_with_sensitive["y_test"],
            predictions_with_sensitive["predictions"],
            y_test_with_sensitive[sensitive_column],
        ),
        "conditional_use_accuracy_equality": conditional_use_accuracy_equality(
            y_test_with_sensitive["y_test"],
            predictions_with_sensitive["predictions"],
            y_test_with_sensitive[sensitive_column],
        ),
        "equal_negative_predictive_value": equal_negative_predictive_value(
            y_test_with_sensitive["y_test"],
            predictions_with_sensitive["predictions"],
            y_test_with_sensitive[sensitive_column],
        ),
        "equal_opportunity": equal_opportunity(
            y_test_with_sensitive["y_test"],
            predictions_with_sensitive["predictions"],
            y_test_with_sensitive[sensitive_column],
        ),
        "overall_accuracy_equality": overall_accuracy_equality(
            y_test_with_sensitive["y_test"],
            predictions_with_sensitive["predictions"],
            y_test_with_sensitive[sensitive_column],
        ),
        "predictive_equality": predictive_equality(
            y_test_with_sensitive["y_test"],
            predictions_with_sensitive["predictions"],
            y_test_with_sensitive[sensitive_column],
        ),
        "statistical_parity": statistical_parity(
            predictions_with_sensitive["predictions"],
            predictions_with_sensitive[sensitive_column],
        ),
        "treatment_equality": treatment_equality(
            y_test_with_sensitive["y_test"],
            predictions_with_sensitive["predictions"],
            y_test_with_sensitive[sensitive_column],
        ),
        "balance_for_positive_class": balance_for_positive_class(
            y_test_with_sensitive["y_test"],
            predictions_with_sensitive["predictions"],
            y_test_with_sensitive[sensitive_column],
        ),
        "equalized_odds": equalized_odds(
            y_test_with_sensitive["y_test"],
            predictions_with_sensitive["predictions"],
            y_test_with_sensitive[sensitive_column],
        ),
        "balance_for_negative_class": balance_for_negative_class(
            y_test_with_sensitive["y_test"],
            predictions_with_sensitive["predictions"],
            y_test_with_sensitive[sensitive_column],
        ),
        "test_fairness": test_fairness(
            y_test_with_sensitive["y_test"],
            predictions_with_sensitive["predictions"],
            y_test_with_sensitive[sensitive_column],
        ),
        "well_calibration": well_calibration(
            y_test_with_sensitive["y_test"],
            predictions_with_sensitive["predictions"],
            y_test_with_sensitive[sensitive_column],
        ),
    }

    return metrics
