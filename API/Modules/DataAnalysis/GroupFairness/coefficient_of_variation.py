import numpy as np


def coefficient_of_variation(y_pred):
    """Coefficient of Variation (CV)"""
    std_dev = np.std(y_pred)
    mean = np.mean(y_pred)
    return std_dev / mean if mean != 0 else 0
