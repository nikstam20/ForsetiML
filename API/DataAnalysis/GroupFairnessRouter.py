from flask import Blueprint, request, jsonify
import pandas as pd
import numpy as np
from Modules.DataAnalysis.GroupFairness.mutual_information_bias import (
    mutual_information,
)
from Modules.DataAnalysis.GroupFairness.consistency import consistency

import seaborn as sns

group_metrics_bp = Blueprint("group", __name__)


def disparate_impact(y_true, sensitive_attr, privileged_group):
    privileged_mask = sensitive_attr == privileged_group
    unprivileged_mask = ~privileged_mask
    prob_privileged = y_true[privileged_mask].mean()
    prob_unprivileged = y_true[unprivileged_mask].mean()
    if prob_unprivileged == 0:
        return 0
    return prob_unprivileged / prob_privileged


def group_mean_difference(y_true, sensitive_attr):
    privileged_group = sensitive_attr.unique()[0]
    privileged_mask = sensitive_attr == privileged_group
    unprivileged_mask = ~privileged_mask
    mean_privileged = y_true[privileged_mask].mean()
    mean_unprivileged = y_true[unprivileged_mask].mean()
    return abs(mean_unprivileged - mean_privileged)


def coefficient_of_variation(y_pred, sensitive_attr):
    groups = sensitive_attr.unique()
    group_cvs = []
    for group in groups:
        group_mask = sensitive_attr == group
        group_values = y_pred[group_mask]
        mean = np.mean(group_values)
        std_dev = np.std(group_values)
        if mean == 0:
            cv = 0
        else:
            cv = std_dev / mean
        scaled_cv = cv / (1 + cv)
        group_cvs.append(scaled_cv)

    overall_cv = np.mean(group_cvs)
    return overall_cv


@group_metrics_bp.route("/metrics", methods=["POST"])
def calculate_new_group_metrics():
    """
    Endpoint to calculate new group metrics dynamically.
    """
    try:
        data = request.json.get("data")
        sensitive_attr = request.json.get("sensitive_attr")
        target_attr = request.json.get("target_attr")
        privileged_group = request.json.get("privileged_group")
        privileged_group = 1
        unprivileged_group = request.json.get("unprivileged_group")
        unprivileged_group = 0

        if privileged_group is None or unprivileged_group is None:
            return (
                jsonify(
                    {"error": "Privileged and unprivileged groups must be specified"}
                ),
                400,
            )

        if not data or not sensitive_attr or not target_attr:
            return (
                jsonify(
                    {
                        "error": "File path, sensitive attribute, and target attribute are required"
                    }
                ),
                400,
            )

        df = pd.read_csv(data)

        if df[target_attr].dtype == "object":
            unique_values = df[target_attr].dropna().unique()
            if len(unique_values) == 2:
                mapping = {val: idx for idx, val in enumerate(unique_values)}
                df[target_attr] = df[target_attr].map(mapping).astype(int)
            else:
                raise ValueError(f"Target attribute '{target_attr}' must be binary.")
        elif not np.issubdtype(df[target_attr].dtype, np.number):
            raise ValueError(
                f"Target attribute '{target_attr}' must be binary or numeric."
            )

        if df[sensitive_attr].dtype == "object":
            df[sensitive_attr] = df[sensitive_attr].astype("category").cat.codes
        elif not np.issubdtype(df[sensitive_attr].dtype, np.number):
            raise ValueError(
                f"Sensitive attribute '{sensitive_attr}' must be categorical or numeric."
            )

        df = df.dropna()

        y_true = df[target_attr].astype(float)
        y_pred = y_true
        X = df.drop(columns=[target_attr, sensitive_attr])

        metrics = {}

        try:
            metrics["Disparate Impact"] = disparate_impact(
                y_true, df[sensitive_attr], privileged_group
            )
            metrics["Group Mean Difference"] = group_mean_difference(
                y_true, df[sensitive_attr]
            )
            metrics["Mutual Information"] = mutual_information(
                y_true, df[sensitive_attr]
            )
            metrics["Consistency"] = consistency(y_true, X)
            metrics["Coefficient of Variation"] = coefficient_of_variation(
                y_pred, df[sensitive_attr]
            )
            total_samples = len(df)
            privileged_count = len(df[df[sensitive_attr] == privileged_group])
            unprivileged_count = len(df[df[sensitive_attr] == unprivileged_group])

            metrics["Sampling Parity Privileged"] = privileged_count / total_samples
            metrics["Sampling Parity Unprivileged"] = unprivileged_count / total_samples
        except Exception as e:
            print("Error calculating metrics:", e)

        metrics = str(metrics)
        return jsonify({"metrics": metrics}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@group_metrics_bp.route("/pdf", methods=["POST"])
def calculate_pdf():
    """
    Endpoint to calculate Probability Density Function (PDF) for sensitive and target attributes.
    """
    try:
        data = request.json.get("data")
        sensitive_attr = request.json.get("sensitive_attr")
        target_attr = request.json.get("target_attr")

        if not data or not sensitive_attr or not target_attr:
            return (
                jsonify(
                    {
                        "error": "File path, sensitive attribute, and target attribute are required"
                    }
                ),
                400,
            )

        df = pd.read_csv(data)

        if sensitive_attr not in df.columns or target_attr not in df.columns:
            return jsonify({"error": "Invalid sensitive or target attribute"}), 400

        df = df[[sensitive_attr, target_attr]].dropna()

        sensitive_pdf = (
            sns.kdeplot(df[sensitive_attr], bw_adjust=0.5).get_lines()[0].get_data()
        )
        target_pdf = (
            sns.kdeplot(df[target_attr], bw_adjust=0.5).get_lines()[0].get_data()
        )

        response_data = {
            "sensitive_pdf": {
                "x": sensitive_pdf[0].tolist(),
                "y": sensitive_pdf[1].tolist(),
            },
            "target_pdf": {
                "x": target_pdf[0].tolist(),
                "y": target_pdf[1].tolist(),
            },
        }

        return jsonify(response_data), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
