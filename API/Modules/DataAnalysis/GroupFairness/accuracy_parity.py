import numpy as np


def accuracy_parity(y_true, y_pred, sensitive_feature):
    """Accuracy Parity"""
    groups = np.unique(sensitive_feature)
    accuracies = {
        group: (
            y_pred[sensitive_feature == group] == y_true[sensitive_feature == group]
        ).mean()
        for group in groups
    }
    return min(accuracies.values()) / max(accuracies.values())
