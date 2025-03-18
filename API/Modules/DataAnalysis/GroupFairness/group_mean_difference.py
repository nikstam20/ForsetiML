import numpy as np


def group_mean_difference(y_pred, sensitive_feature):
    """Group Mean Difference"""
    groups = np.unique(sensitive_feature)
    mean_values = {
        group: np.mean(y_pred[sensitive_feature == group]) for group in groups
    }
    gmd = mean_values[groups[1]] - mean_values[groups[0]]
    return gmd
