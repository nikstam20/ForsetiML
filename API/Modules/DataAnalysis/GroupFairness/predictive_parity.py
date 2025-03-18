def predictive_parity(y_true, y_pred, sensitive_feature):
    """
    Compute Predictive Parity.
    Checks if Positive Predictive Value (PPV) is equal across sensitive groups.

    Args:
        y_true (pd.Series): Ground truth outcomes.
        y_pred (pd.Series): Predicted outcomes.
        sensitive_feature (pd.Series): Sensitive attribute.

    Returns:
        dict: Dictionary containing PPV for each group.
    """
    groups = sensitive_feature.unique()
    ppv_dict = {}

    for group in groups:
        mask = sensitive_feature == group
        true_positives = ((y_pred[mask] == 1) & (y_true[mask] == 1)).sum()
        predicted_positives = (y_pred[mask] == 1).sum()

        ppv = true_positives / predicted_positives if predicted_positives > 0 else 0
        ppv_dict[group] = ppv

    return ppv_dict
