import numpy as np


def bounded_group_loss(y_true, sensitive_attr):
    """
    Calculate Bounded Group Loss.

    Args:
        y_true (array-like): True labels or predicted probabilities.
        sensitive_attr (array-like): Sensitive attribute for grouping.

    Returns:
        float: Difference in loss across groups.
    """
    unique_groups = np.unique(sensitive_attr)
    group_losses = [
        np.mean(1 - y_true[sensitive_attr == group]) for group in unique_groups
    ]  # Example loss: (1 - y_true)

    return max(group_losses) - min(group_losses)
