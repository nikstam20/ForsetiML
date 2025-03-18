import numpy as np


def generalized_entropy_index(y_pred, alpha=2):
    """Generalized Entropy Index (GEI)"""
    mean = np.mean(y_pred)
    gei_values = np.mean(((y_pred / mean) ** alpha) - 1) / (alpha * (alpha - 1))
    print(gei_values)
    return gei_values
