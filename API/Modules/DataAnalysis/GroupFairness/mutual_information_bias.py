from sklearn.metrics import mutual_info_score


def mutual_information(y_true, sensitive_feature):
    """
    Compute Mutual Information.
    Measures the statistical dependence between sensitive attribute and outcome.

    Args:
        y_true (pd.Series): Ground truth outcomes.
        sensitive_feature (pd.Series): Sensitive attribute.

    Returns:
        float: Mutual information score.
    """
    return mutual_info_score(y_true, sensitive_feature)
