from sklearn.metrics import accuracy_score
import numpy as np

def pairwise_accuracy_diff(y_true, y_pred, sensitive_attr):
    """
    Calculate Pairwise Equal Accuracy Difference between groups.

    Args:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.
        sensitive_attr (array-like): Sensitive attribute for grouping.

    Returns:
        float: Maximum accuracy difference between any two groups.
    """
    unique_groups = np.unique(sensitive_attr)
    accuracies = {group: accuracy_score(y_true[sensitive_attr == group], y_pred[sensitive_attr == group])
                  for group in unique_groups}

    max_diff = max(abs(acc1 - acc2) for g1, acc1 in accuracies.items() for g2, acc2 in accuracies.items() if g1 != g2)
    return max_diff
