import numpy as np


def theil_index(y_pred):
    """Theil Index (Special case of GEI with alpha = 1)"""
    mean = np.mean(y_pred)
    theil_values = np.mean((y_pred / mean) * np.log(y_pred / mean))
    return theil_values
