import numpy as np


def group_calibration(y_true, y_pred_proba, sensitive_feature, bins=10):
    """
    Compute Group Calibration.
    Measures whether predicted probabilities match observed outcomes equally across groups.

    Args:
        y_true (pd.Series): Ground truth outcomes.
        y_pred_proba (pd.Series): Predicted probabilities.
        sensitive_feature (pd.Series): Sensitive attribute.
        bins (int): Number of bins to discretize probabilities.

    Returns:
        dict: Calibration results per group.
    """
    groups = sensitive_feature.unique()
    calibration = {}

    for group in groups:
        mask = sensitive_feature == group
        y_true_group = y_true[mask]
        y_pred_proba_group = y_pred_proba[mask]

        bin_edges = np.linspace(0, 1, bins + 1)
        bin_indices = np.digitize(y_pred_proba_group, bin_edges, right=True)
        calibration[group] = []

        for i in range(1, bins + 1):
            bin_mask = bin_indices == i
            if bin_mask.sum() > 0:
                observed = y_true_group[bin_mask].mean()
                predicted = y_pred_proba_group[bin_mask].mean()
                calibration[group].append((observed, predicted))

    return calibration
