from flask import Blueprint, request, jsonify
import pandas as pd
import os
import pickle
import numpy as np
from sklearn.metrics import r2_score
from xgboost import XGBClassifier, XGBRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
)
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
from Modules.DataAnalysis.GroupFairness.notions_regression import (
    disparate_impact_regression,
    conditional_accuracy_equality_regression,
    conditional_statistical_parity_regression,
    equal_opportunity_regression,
    predictive_equality_regression,
    treatment_equality_regression,
    balance_for_positive_class_regression,
    balance_for_negative_class_regression,
    equalized_odds_regression,
)
from Modules.DataAnalysis.GroupFairness.disparate_impact import disparate_impact
from Modules.DataAnalysis.GroupFairness.coefficient_of_variation import (
    coefficient_of_variation,
)
from Modules.DataAnalysis.GroupFairness.generalized_entropy_index import (
    generalized_entropy_index,
)
from Modules.DataAnalysis.GroupFairness.theil_index import theil_index
from Modules.DataAnalysis.GroupFairness.equal_opportuity_difference import (
    equal_opportunity_difference,
)
from Modules.DataAnalysis.GroupFairness.error_gap import error_rate_difference
from Modules.DataAnalysis.GroupFairness.statistical_parity import (
    statistical_parity_difference,
)
from Modules.DataAnalysis.GroupFairness.ks_stat_parity import ks_statistical_parity
from Modules.DataAnalysis.GroupFairness.group_mean_difference import (
    group_mean_difference,
)
from Modules.DataAnalysis.GroupFairness.auc_parity import auc_parity
from Modules.DataAnalysis.GroupFairness.accuracy_parity import accuracy_parity
from Modules.DataAnalysis.GroupFairness.max_mean_diff import max_mean_difference

model_bp = Blueprint("model", __name__)

MODEL_FILE_CLASSIFICATION = "xgboost_classification_model.pkl"
MODEL_FILE_REGRESSION = "xgboost_regression_model.pkl"
LABEL_ENCODERS_FILE = "label_encoders.pkl"

from neo4j import GraphDatabase


def fetch_metric_limitations():
    uri = "bolt://localhost:7687"
    user = "neo4j"
    password = "password"
    driver = GraphDatabase.driver(uri, auth=(user, password))
    query = """
    MATCH (m:Metric)-[:HAS_LIMITATION]->(l:Limitation)
    RETURN m.name AS MetricName, collect(l.name) AS Limitations
    """
    with driver.session() as session:
        result = session.run(query)
        metric_limitations = {
            record["MetricName"]: record["Limitations"] for record in result
        }
    driver.close()
    return metric_limitations


def save_label_encoders(label_encoders):
    with open(LABEL_ENCODERS_FILE, "wb") as f:
        pickle.dump(label_encoders, f)


def load_label_encoders():
    if not os.path.exists(LABEL_ENCODERS_FILE):
        raise FileNotFoundError(
            "Label encoders file not found. Ensure you have trained the model."
        )
    with open(LABEL_ENCODERS_FILE, "rb") as f:
        return pickle.load(f)


def encode_categorical_columns(data, training=True, encoders=None):
    """
    Encodes categorical columns in the given DataFrame.

    Args:
        data (pd.DataFrame): The data to encode.
        training (bool): Whether this is the training phase (fit and transform).
        encoders (dict): Preloaded encoders to use for transformation during inference.

    Returns:
        pd.DataFrame: Transformed DataFrame with categorical columns encoded.
        dict: Encoders used (if training=True).
    """
    if training:
        encoders = {}
        categorical_cols = data.select_dtypes(include=["object"]).columns
        for col in categorical_cols:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])
            encoders[col] = le
        return data, encoders
    else:
        if not encoders:
            raise ValueError("Encoders must be provided for inference!")
        for col, le in encoders.items():
            if col in data.columns:
                data[col] = le.transform(data[col])
        return data


def save_label_encoders(label_encoders):
    """
    Save label encoders to a file.

    Args:
        label_encoders (dict): Encoders to save.
    """
    with open("label_encoders.pkl", "wb") as f:
        pickle.dump(label_encoders, f)


def save_target_label_encoder(encoder):
    """
    Save the target label encoder to a file.

    Args:
        encoder (LabelEncoder): The target encoder to save.
    """
    with open("target_label_encoder.pkl", "wb") as f:
        pickle.dump(encoder, f)


import time
import os
import json


@model_bp.route("/train_classification_model", methods=["POST"])
def train_classification_model():
    try:
        data_path = request.json.get("data_path")
        target_column = request.json.get("target_column")
        model_params = request.json.get("model_params", {})
        train_split_percent = request.json.get("train_split_percent", 80)
        if not data_path or not target_column:
            return (
                jsonify(
                    {
                        "error": "Missing required parameters 'data_path' or 'target_column'"
                    }
                ),
                400,
            )
        if not os.path.exists(data_path):
            return jsonify({"error": f"File not found at {data_path}"}), 400

        data = pd.read_csv(data_path)
        if target_column not in data.columns:
            return (
                jsonify(
                    {"error": f"Target column '{target_column}' not found in data"}
                ),
                400,
            )
        split_index = int(len(data) * (train_split_percent / 100))
        train_data = data.iloc[:split_index]

        X_train = train_data.drop(columns=[target_column])
        y_train = train_data[target_column]

        target_label_encoder = LabelEncoder()
        y_train = target_label_encoder.fit_transform(y_train)

        X_train, feature_encoders = encode_categorical_columns(X_train, training=True)
        start_time = time.time()
        model = XGBClassifier(**model_params)
        model.fit(X_train, y_train)
        end_time = time.time()

        with open(MODEL_FILE_CLASSIFICATION, "wb") as f:
            pickle.dump(model, f)
        save_label_encoders(feature_encoders)
        save_target_label_encoder(target_label_encoder)
        training_time = end_time - start_time
        model_size = os.path.getsize(MODEL_FILE_CLASSIFICATION)

        return (
            jsonify(
                {
                    "message": "Classification model trained and saved successfully",
                    "training_time_seconds": training_time,
                    "model_size_bytes": model_size,
                }
            ),
            200,
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@model_bp.route("/train_regression_model", methods=["POST"])
def train_regression_model():
    try:
        data_path = request.json.get("data_path")
        target_column = request.json.get("target_column")
        model_params = request.json.get("model_params", {})
        train_split_percent = request.json.get("train_split_percent", 80)

        if not data_path or not target_column:
            return (
                jsonify(
                    {
                        "error": "Missing required parameters 'data_path' or 'target_column'"
                    }
                ),
                400,
            )

        if not os.path.exists(data_path):
            return jsonify({"error": f"File not found at {data_path}"}), 400

        data = pd.read_csv(data_path)

        if target_column not in data.columns:
            return (
                jsonify(
                    {"error": f"Target column '{target_column}' not found in data"}
                ),
                400,
            )

        split_index = int(len(data) * (train_split_percent / 100))
        train_data = data.iloc[:split_index]

        X_train = train_data.drop(columns=[target_column])
        y_train = train_data[target_column]

        if not pd.api.types.is_numeric_dtype(y_train):
            return (
                jsonify(
                    {
                        "error": f"Target column '{target_column}' must be numeric for regression"
                    }
                ),
                400,
            )

        X_train, feature_encoders = encode_categorical_columns(X_train, training=True)

        start_time = time.time()
        model = XGBRegressor(**model_params)
        model.fit(X_train, y_train)
        training_time = time.time() - start_time

        with open(MODEL_FILE_REGRESSION, "wb") as f:
            pickle.dump(model, f)

        save_label_encoders(feature_encoders)
        model_size = os.path.getsize(MODEL_FILE_REGRESSION)

        return (
            jsonify(
                {
                    "message": "Regression model trained and saved successfully",
                    "training_time_seconds": training_time,
                    "model_size_bytes": model_size,
                }
            ),
            200,
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@model_bp.route("/make_classification_inference", methods=["POST"])
def make_classification_inference():
    try:
        test_data_path = request.json.get("test_data_path")
        test_split_percent = request.json.get("test_split_percent", 20)
        target_column = request.json.get("target_column")
        sensitive_feature = request.json.get("sensitive_feature")
        limitations = request.json.get("limitations")

        if not test_data_path or not target_column:
            return (
                jsonify(
                    {
                        "error": "Missing required parameters 'test_data_path' or 'target_column'"
                    }
                ),
                400,
            )

        if not os.path.exists(test_data_path):
            return jsonify({"error": f"File not found at {test_data_path}"}), 400

        test_data = pd.read_csv(test_data_path)

        if not os.path.exists(MODEL_FILE_CLASSIFICATION):
            return (
                jsonify(
                    {
                        "error": "Trained classification model not found. Please train the model first."
                    }
                ),
                400,
            )
        with open(MODEL_FILE_CLASSIFICATION, "rb") as f:
            model = pickle.load(f)
        split_index = int(len(test_data) * ((100 - test_split_percent) / 100))
        test_data = test_data.iloc[split_index:]
        X_test = test_data.drop(columns=[target_column])
        y_test = test_data[target_column]
        with open("label_encoders.pkl", "rb") as f:
            label_encoders = pickle.load(f)

        categorical_cols = X_test.select_dtypes(include=["object"]).columns
        for col in categorical_cols:
            if col in label_encoders:
                le = label_encoders[col]
                X_test[col] = le.transform(X_test[col])
            else:
                raise ValueError(
                    f"Encoder for column '{col}' is missing in label_encoders.pkl."
                )

        with open("target_label_encoder.pkl", "rb") as f:
            target_label_encoder = pickle.load(f)
        y_test = target_label_encoder.transform(y_test)
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)

        tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions)
        recall = recall_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)
        sensitive_feature_column = test_data[sensitive_feature]

        y_test_with_sensitive = pd.DataFrame(
            {"y_test": y_test, f"{sensitive_feature}": sensitive_feature_column}
        )

        predictions_with_sensitive = pd.DataFrame(
            {
                "predictions": predictions,
                f"{sensitive_feature}": sensitive_feature_column,
            }
        )

        metric_limitations = fetch_metric_limitations()

        def can_use_metric(metric_name):
            metric_issues = metric_limitations.get(metric_name, [])
            return not any(issue in limitations for issue in metric_issues)

        statistical_parity_metric = disparate_impact(
            y_test_with_sensitive["y_test"],
            predictions_with_sensitive["predictions"],
            y_test_with_sensitive[sensitive_feature],
        )
        if can_use_metric("Error Rate Difference"):
            conditional_accuracy_eq = error_rate_difference(
                y_test_with_sensitive["y_test"],
                predictions_with_sensitive["predictions"],
                y_test_with_sensitive[sensitive_feature],
            )
        elif can_use_metric("ΑUC Parity"):
            conditional_accuracy_eq = auc_parity(
                y_test_with_sensitive["y_test"],
                predictions_with_sensitive["predictions"],
                y_test_with_sensitive[sensitive_feature],
            )
        else:
            conditional_accuracy_eq = conditional_use_accuracy_equality(
                y_test_with_sensitive["y_test"],
                predictions_with_sensitive["predictions"],
                y_test_with_sensitive[sensitive_feature],
            )

        conditional_stat_parity = conditional_statistical_parity(
            y_test_with_sensitive["y_test"],
            predictions_with_sensitive["predictions"],
            y_test_with_sensitive[sensitive_feature],
        )

        if can_use_metric("Equal Opportunity Difference"):
            equal_opportunity_metric = equal_opportunity_difference(
                y_test_with_sensitive["y_test"],
                predictions_with_sensitive["predictions"],
                y_test_with_sensitive[sensitive_feature],
            )
        elif can_use_metric("ΑUC Parity"):
            equal_opportunity_metric = auc_parity(
                y_test_with_sensitive["y_test"],
                predictions_with_sensitive["predictions"],
                y_test_with_sensitive[sensitive_feature],
            )
        else:
            equal_opportunity_metric = equal_opportunity(
                y_test_with_sensitive["y_test"],
                predictions_with_sensitive["predictions"],
                y_test_with_sensitive[sensitive_feature],
            )

        equal_neg_predictive_val = equal_negative_predictive_value(
            y_test_with_sensitive["y_test"],
            predictions_with_sensitive["predictions"],
            y_test_with_sensitive[sensitive_feature],
        )

        if can_use_metric("Accuracy Parity"):
            overall_accuracy_eq = accuracy_parity(
                y_test_with_sensitive["y_test"],
                predictions_with_sensitive["predictions"],
                y_test_with_sensitive[sensitive_feature],
            )
        else:
            overall_accuracy_eq = overall_accuracy_equality(
                y_test_with_sensitive["y_test"],
                predictions_with_sensitive["predictions"],
                y_test_with_sensitive[sensitive_feature],
            )

        predictive_eq = predictive_equality(
            y_test_with_sensitive["y_test"],
            predictions_with_sensitive["predictions"],
            y_test_with_sensitive[sensitive_feature],
        )

        treatment_eq = treatment_equality(
            y_test_with_sensitive["y_test"],
            predictions_with_sensitive["predictions"],
            y_test_with_sensitive[sensitive_feature],
        )

        balance_positive_class = balance_for_positive_class(
            y_test_with_sensitive["y_test"],
            predictions_with_sensitive["predictions"],
            y_test_with_sensitive[sensitive_feature],
        )

        balance_negative_class = balance_for_negative_class(
            y_test_with_sensitive["y_test"],
            predictions_with_sensitive["predictions"],
            y_test_with_sensitive[sensitive_feature],
        )

        if can_use_metric("Group Mean Difference"):
            equalized_odds_metric = group_mean_difference(
                predictions_with_sensitive["predictions"],
                y_test_with_sensitive[sensitive_feature],
            )
        elif can_use_metric("Max Mean Difference"):
            equalized_odds_metric = max_mean_difference(
                y_test_with_sensitive["y_test"],
                predictions_with_sensitive["predictions"],
                y_test_with_sensitive[sensitive_feature],
            )
        elif can_use_metric("Error Rate Difference"):
            equalized_odds_metric = error_rate_difference(
                y_test_with_sensitive["y_test"],
                predictions_with_sensitive["predictions"],
                y_test_with_sensitive[sensitive_feature],
            )
        else:
            equalized_odds_metric = equalized_odds(
                y_test_with_sensitive["y_test"],
                predictions_with_sensitive["predictions"],
                y_test_with_sensitive[sensitive_feature],
            )

        equal_neg_predictive_val = equal_negative_predictive_value(
            y_test_with_sensitive["y_test"],
            predictions_with_sensitive["predictions"],
            y_test_with_sensitive[sensitive_feature],
        )

        test_fairness_metric = test_fairness(
            y_test_with_sensitive["y_test"],
            predictions_with_sensitive["predictions"],
            y_test_with_sensitive[sensitive_feature],
            probabilities,
        )

        well_calibration_metric = well_calibration(
            y_test_with_sensitive["y_test"],
            predictions_with_sensitive["predictions"],
            y_test_with_sensitive[sensitive_feature],
            probabilities,
        )

        metrics = {
            "predictions": predictions.tolist(),
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
            "conditional_statistical_parity": abs(conditional_stat_parity),
            "conditional_use_accuracy_equality": abs(conditional_accuracy_eq),
            "equal_negative_predictive_value": abs(equal_neg_predictive_val),
            "equal_opportunity": abs(equal_opportunity_metric),
            "overall_accuracy_equality": abs(overall_accuracy_eq),
            "predictive_equality": abs(predictive_eq),
            "statistical_parity": abs(statistical_parity_metric),
            "treatment_equality": abs(treatment_eq),
            "balance_for_positive_class": abs(balance_positive_class),
            "equalized_odds": abs(equalized_odds_metric),
            "balance_for_negative_class": abs(balance_negative_class),
            "test_fairness": abs(test_fairness_metric),
            "well_calibration": abs(well_calibration_metric),
        }

        return jsonify(metrics), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@model_bp.route("/make_regression_inference", methods=["POST"])
def make_regression_inference():
    try:
        test_data_path = request.json.get("test_data_path")
        test_split_percent = request.json.get("test_split_percent", 20)
        target_column = request.json.get("target_column")
        sensitive_feature = request.json.get("sensitive_feature")
        limitations = request.json.get("limitations")

        if not test_data_path or not target_column:
            return (
                jsonify(
                    {
                        "error": "Missing required parameters 'test_data_path' or 'target_column'"
                    }
                ),
                400,
            )

        if not os.path.exists(test_data_path):
            return jsonify({"error": f"File not found at {test_data_path}"}), 400

        if not os.path.exists(MODEL_FILE_REGRESSION):
            return (
                jsonify(
                    {
                        "error": "Trained regression model not found. Please train the model first."
                    }
                ),
                400,
            )

        with open(MODEL_FILE_REGRESSION, "rb") as f:
            model = pickle.load(f)

        test_data = pd.read_csv(test_data_path)
        split_index = int(len(test_data) * ((100 - test_split_percent) / 100))
        test_data = test_data.iloc[split_index:]
        X_test = test_data.drop(columns=[target_column])
        y_test = test_data[target_column]

        predictions = model.predict(X_test)
        sensitive_feature_column = test_data[sensitive_feature]

        y_test_with_sensitive = pd.DataFrame(
            {"y_test": y_test, sensitive_feature: sensitive_feature_column}
        ).reset_index(drop=True)

        predictions_with_sensitive = pd.DataFrame(
            {"predictions": predictions, sensitive_feature: sensitive_feature_column}
        ).reset_index(drop=True)
        results = {
            "mse": float(mean_squared_error(y_test, predictions)),
            "mae": float(mean_absolute_error(y_test, predictions)),
            "r2_score": float(r2_score(y_test, predictions)),
            "statistical_parity": disparate_impact_regression(
                y_test_with_sensitive, predictions_with_sensitive, sensitive_feature
            ),
            "conditional_accuracy_equality": conditional_accuracy_equality_regression(
                y_test_with_sensitive, predictions_with_sensitive, sensitive_feature
            ),
            "conditional_statistical_parity": conditional_statistical_parity_regression(
                y_test_with_sensitive, predictions_with_sensitive, sensitive_feature
            ),
            "equal_opportunity": equal_opportunity_regression(
                y_test_with_sensitive, predictions_with_sensitive, sensitive_feature
            ),
            "predictive_equality": predictive_equality_regression(
                y_test_with_sensitive, predictions_with_sensitive, sensitive_feature
            ),
            "treatment_equality": treatment_equality_regression(
                y_test_with_sensitive, predictions_with_sensitive, sensitive_feature
            ),
            "balance_for_positive_class": balance_for_positive_class_regression(
                y_test_with_sensitive, predictions_with_sensitive, sensitive_feature
            ),
            "balance_for_negative_class": balance_for_negative_class_regression(
                y_test_with_sensitive, predictions_with_sensitive, sensitive_feature
            ),
            "equalized_odds": equalized_odds_regression(
                y_test_with_sensitive, predictions_with_sensitive, sensitive_feature
            ),
        }

        for key, value in results.items():
            if isinstance(value, np.floating):
                results[key] = float(value)
        return jsonify(results), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
