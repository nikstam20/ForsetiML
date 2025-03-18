import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def disparate_impact_regression(y_true, y_pred, sensitive_column):
    if isinstance(y_pred, pd.DataFrame):
        y_pred = y_pred["predictions"]
    groups = y_true.groupby(sensitive_column)
    means = groups.apply(lambda x: np.mean(y_pred.iloc[x.index]))
    result = means.min() / means.max() if means.max() > 0 else 0
    return result


def conditional_accuracy_equality_regression(y_true, y_pred, sensitive_column):
    if isinstance(y_pred, pd.DataFrame):
        y_pred = y_pred["predictions"]
    groups = y_true.groupby(sensitive_column)
    mse = groups.apply(
        lambda x: mean_squared_error(
            y_true["y_test"].iloc[x.index], y_pred.iloc[x.index]
        )
    )
    result = mse.min() / mse.max() if mse.max() > 0 else 0
    return result


def conditional_statistical_parity_regression(y_true, y_pred, sensitive_column):
    data = y_true.join(y_pred, rsuffix="_pred")
    data["residuals"] = data["y_test"] - data["predictions"]
    groups = data.groupby(sensitive_column)["residuals"]
    means = abs(groups.mean())
    result = means.min() / means.max() if means.max() > 0 else 0
    return result


def equal_opportunity_regression(y_true, y_pred, sensitive_column):
    if isinstance(y_pred, pd.DataFrame):
        y_pred = y_pred["predictions"]
    groups = y_true.groupby(sensitive_column)
    r2 = groups.apply(
        lambda x: r2_score(y_true["y_test"].iloc[x.index], y_pred.iloc[x.index])
    )
    result = r2.min() / r2.max() if r2.max() > 0 else 0
    return result


def predictive_equality_regression(y_true, y_pred, sensitive_column):
    data = y_true.join(y_pred, rsuffix="_pred")
    data["residuals"] = data["y_test"] - data["predictions"]
    groups = data.groupby(sensitive_column)["residuals"]
    variances = groups.var()
    result = variances.min() / variances.max() if variances.max() > 0 else 0
    return result


def treatment_equality_regression(y_true, y_pred, sensitive_column):
    data = y_true.join(y_pred, rsuffix="_pred")
    data["under_predictions"] = (data["predictions"] < data["y_test"]).astype(int)
    data["over_predictions"] = (data["predictions"] >= data["y_test"]).astype(int)

    groups = data.groupby(sensitive_column)["under_predictions"]
    under_ratio = groups.mean()
    groups = data.groupby(sensitive_column)["over_predictions"]
    over_ratio = groups.mean()
    result = (
        (under_ratio / over_ratio).min() / (under_ratio / over_ratio).max()
        if (under_ratio / over_ratio).max() > 0
        else 0
    )
    return result


def balance_for_positive_class_regression(y_true, y_pred, sensitive_column):
    data = y_true.join(y_pred, rsuffix="_pred")
    data["above_average"] = (data["predictions"] > data["y_test"].mean()).astype(int)
    groups = data.groupby(sensitive_column)["above_average"]
    balance = groups.mean()
    result = balance.min() / balance.max() if balance.max() > 0 else 0
    return result


def balance_for_negative_class_regression(y_true, y_pred, sensitive_column):
    data = y_true.join(y_pred, rsuffix="_pred")
    data["above_average"] = (data["predictions"] <= data["y_test"].mean()).astype(int)
    groups = data.groupby(sensitive_column)["above_average"]
    balance = groups.mean()
    result = balance.min() / balance.max() if balance.max() > 0 else 0
    return result


def equalized_odds_regression(y_true, y_pred, sensitive_column):
    predictive_eq = predictive_equality_regression(y_true, y_pred, sensitive_column)
    equal_opp = equal_opportunity_regression(y_true, y_pred, sensitive_column)
    eq_odds = (predictive_eq + equal_opp) / 2
    return eq_odds


def total(y_true, y_pred, sensitive_column):
    results = {
        "statistical_parity": disparate_impact_regression(
            y_true, y_pred, sensitive_column
        ),
        "conditional_accuracy_equality": conditional_accuracy_equality_regression(
            y_true, y_pred, sensitive_column
        ),
        "conditional_statistical_parity": conditional_statistical_parity_regression(
            y_true, y_pred, sensitive_column
        ),
        "equal_opportunity": equal_opportunity_regression(
            y_true, y_pred, sensitive_column
        ),
        "predictive_equality": predictive_equality_regression(
            y_true, y_pred, sensitive_column
        ),
        "treatment_equality": treatment_equality_regression(
            y_true, y_pred, sensitive_column
        ),
        "balance_for_positive_class": balance_for_positive_class_regression(
            y_true, y_pred, sensitive_column
        ),
        "balance_for_negative_class": balance_for_negative_class_regression(
            y_true, y_pred, sensitive_column
        ),
        "equalized_odds": equalized_odds_regression(y_true, y_pred, sensitive_column),
    }
    return results
