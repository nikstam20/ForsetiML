import numpy as np


def fairness_degree(y_true, sensitive_feature):
    """
    Calculate Fairness Degree.

    Args:
        y_true: Ground truth or predicted outcomes (binary).
        sensitive_feature: Sensitive attribute for grouping.

    Returns:
        dict: Fairness degree (proportion of favorable outcomes) for each group.
    """
    unique_groups = np.unique(sensitive_feature)
    fairness_degrees = {}

    for group in unique_groups:
        group_mask = sensitive_feature == group
        favorable_outcomes = np.sum(y_true[group_mask] == 1)
        total_outcomes = np.sum(group_mask)
        fairness_degrees[group] = (
            favorable_outcomes / total_outcomes if total_outcomes > 0 else 0
        )

    return fairness_degrees
