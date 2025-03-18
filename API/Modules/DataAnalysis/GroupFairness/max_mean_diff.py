import numpy as np


def max_mean_difference(y_pred, sensitive_feature):
    """Max Mean Difference"""
    groups = np.unique(sensitive_feature)
    group_means = [np.mean(y_pred[sensitive_feature == group]) for group in groups]
    return min(group_means) / max(group_means)
