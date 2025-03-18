def average_odds_difference(y_true, y_pred, sensitive_feature):
    """
    Compute Average Odds Difference.
    Measures the difference in average TPR and FPR between groups.

    Args:
        y_true (pd.Series): Ground truth outcomes.
        y_pred (pd.Series): Predicted outcomes.
        sensitive_feature (pd.Series): Sensitive attribute.

    Returns:
        float: Average odds difference between groups.
    """
    groups = sensitive_feature.unique()
    group_metrics = {}

    for group in groups:
        mask = sensitive_feature == group
        true_positives = ((y_pred[mask] == 1) & (y_true[mask] == 1)).sum()
        false_positives = ((y_pred[mask] == 1) & (y_true[mask] == 0)).sum()
        positives = (y_true[mask] == 1).sum()
        negatives = (y_true[mask] == 0).sum()

        tpr = true_positives / positives if positives > 0 else 0
        fpr = false_positives / negatives if negatives > 0 else 0
        group_metrics[group] = {"TPR": tpr, "FPR": fpr}

    tpr_diff = abs(group_metrics[groups[0]]["TPR"] - group_metrics[groups[1]]["TPR"])
    fpr_diff = abs(group_metrics[groups[0]]["FPR"] - group_metrics[groups[1]]["FPR"])

    return (tpr_diff + fpr_diff) / 2
