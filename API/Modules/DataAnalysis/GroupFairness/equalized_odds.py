def equalized_odds(y_true, y_pred, sensitive_feature):
    """
    Compute Equalized Odds.
    Checks if True Positive Rate (TPR) and False Positive Rate (FPR) are equal across sensitive groups.

    Args:
        y_true (pd.Series): Ground truth outcomes.
        y_pred (pd.Series): Predicted outcomes.
        sensitive_feature (pd.Series): Sensitive attribute.

    Returns:
        dict: Dictionary containing TPR and FPR for each group.
    """
    groups = sensitive_feature.unique()
    metrics = {}

    for group in groups:
        mask = sensitive_feature == group
        true_positives = ((y_pred[mask] == 1) & (y_true[mask] == 1)).sum()
        false_positives = ((y_pred[mask] == 1) & (y_true[mask] == 0)).sum()
        positives = (y_true[mask] == 1).sum()
        negatives = (y_true[mask] == 0).sum()

        tpr = true_positives / positives if positives > 0 else 0
        fpr = false_positives / negatives if negatives > 0 else 0

        metrics[group] = {"TPR": tpr, "FPR": fpr}

    return metrics
