def equality_of_opportunity(y_true, y_pred, sensitive_feature):
    """
    Compute Equality of Opportunity.
    Checks if True Positive Rate (TPR) is equal across sensitive groups.

    Args:
        y_true (pd.Series): Ground truth outcomes.
        y_pred (pd.Series): Predicted outcomes.
        sensitive_feature (pd.Series): Sensitive attribute.

    Returns:
        dict: Dictionary containing TPR for each group.
    """
    groups = sensitive_feature.unique()
    tpr_dict = {}

    for group in groups:
        mask = sensitive_feature == group
        true_positives = ((y_pred[mask] == 1) & (y_true[mask] == 1)).sum()
        positives = (y_true[mask] == 1).sum()

        tpr = true_positives / positives if positives > 0 else 0
        tpr_dict[group] = tpr

    return tpr_dict
