import numpy as np


def equal_opportunity_difference(y_true, y_pred, sensitive_feature):
    """Equal Opportunity Difference"""
    groups = np.unique(sensitive_feature)
    tpr_values = {
        group: (
            y_pred[sensitive_feature == group] == y_true[sensitive_feature == group]
        ).mean()
        for group in groups
    }
    print(tpr_values)
    if tpr_values[groups[0]] > tpr_values[groups[1]]:
        eod = tpr_values[groups[1]] / tpr_values[groups[0]]
    else:
        eod = tpr_values[groups[0]] / tpr_values[groups[1]]
    return eod
