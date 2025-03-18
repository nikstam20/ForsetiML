import numpy as np


def statistical_parity_difference(y_pred, sensitive_feature):
    """Statistical Parity Difference"""
    groups = np.unique(sensitive_feature)
    positive_rates = {
        group: (y_pred[sensitive_feature == group] == 1).mean() for group in groups
    }
    spd = positive_rates[groups[1]] - positive_rates[groups[0]]
    return spd
