import numpy as np
from sklearn.metrics import roc_auc_score


def auc_parity(y_true, y_pred, sensitive_feature):
    """AUC Parity"""
    groups = np.unique(sensitive_feature)
    auc_values = {
        group: roc_auc_score(
            y_true[sensitive_feature == group],
            y_pred[sensitive_feature == group],
            average="micro",
        )
        for group in groups
    }
    max_auc = max(auc_values.values())
    min_auc = min(auc_values.values())
    return min_auc / max_auc
