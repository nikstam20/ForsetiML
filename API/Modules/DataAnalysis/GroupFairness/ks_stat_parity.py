from scipy.stats import ks_2samp
import numpy as np


def ks_statistical_parity(y_pred, sensitive_feature):
    """KS Statistical Parity using the Kolmogorov-Smirnov test."""
    groups = np.unique(sensitive_feature)
    distributions = [y_pred[sensitive_feature == group] for group in groups]
    ks_stat, _ = ks_2samp(distributions[0], distributions[1])
    return ks_stat
