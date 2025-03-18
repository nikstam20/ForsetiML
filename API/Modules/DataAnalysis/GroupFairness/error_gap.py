import numpy as np


def error_rate_difference(y_true, y_pred, sensitive_feature):
    """Error Rate Difference"""
    groups = np.unique(sensitive_feature)
    error_rates = {
        group: 1
        - (
            y_pred[sensitive_feature == group] == y_true[sensitive_feature == group]
        ).mean()
        for group in groups
    }
    if error_rates[groups[0]] > error_rates[groups[1]]:
        erd = error_rates[groups[1]] / error_rates[groups[0]]
    else:
        erd = error_rates[groups[0]] / error_rates[groups[1]]
    return erd
