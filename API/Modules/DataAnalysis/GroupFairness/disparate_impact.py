import numpy as np


def disparate_impact(y_true, y_pred, sensitive_feature):
    """Disparate Impact"""
    groups = np.unique(sensitive_feature)
    positive_rates = {
        group: (y_pred[sensitive_feature == group] == 1).mean() for group in groups
    }
    if positive_rates[groups[0]] > positive_rates[groups[1]]:
        di_values = (
            positive_rates[groups[1]] / positive_rates[groups[0]]
            if positive_rates[groups[0]] != 0
            else 0
        )
    else:
        di_values = (
            positive_rates[groups[0]] / positive_rates[groups[1]]
            if positive_rates[groups[1]] != 0
            else 0
        )
    return di_values
