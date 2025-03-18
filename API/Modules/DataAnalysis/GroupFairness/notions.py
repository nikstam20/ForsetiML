import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    accuracy_score,
    brier_score_loss,
)


def conditional_statistical_parity(y_true, y_pred, sensitive_feature):
    """Conditional Statistical Parity"""
    groups = np.unique(sensitive_feature)
    group_probs = {
        group: np.mean(y_pred[sensitive_feature == group]) for group in groups
    }
    return (
        min(group_probs.values()) / max(group_probs.values())
        if max(group_probs.values()) > 0
        else 0
    )


def equal_negative_predictive_value(y_true, y_pred, sensitive_feature):
    """Equal Negative Predictive Value"""
    groups = np.unique(sensitive_feature)
    group_counts = {group: len(y_true[sensitive_feature == group]) for group in groups}

    total_product = np.prod(list(group_counts.values()))
    tt = len(y_true)

    npv_values = {}
    for group in groups:
        tn, fp, fn, tp = confusion_matrix(
            y_true[sensitive_feature == group], y_pred[sensitive_feature == group]
        ).ravel()
        npv_values[group] = (
            (tn / (tn + fn)) * (total_product / group_counts[group])
            if (tn + fn) > 0
            else 0
        )
    return (
        min(npv_values.values()) / max(npv_values.values())
        if max(npv_values.values()) > 0
        else 0
    )


def equal_opportunity(y_true, y_pred, sensitive_feature):
    """Equal Opportunity (True Positive Rate Parity)"""
    groups = np.unique(sensitive_feature)
    group_counts = {group: len(y_true[sensitive_feature == group]) for group in groups}

    total_product = np.prod(list(group_counts.values()))
    tt = len(y_true)
    tpr_values = {}
    for group in groups:
        tn, fp, fn, tp = confusion_matrix(
            y_true[sensitive_feature == group], y_pred[sensitive_feature == group]
        ).ravel()
        tpr_values[group] = (tp / (fn + tp)) if (fn + tp) > 0 else 0

    return (
        min(tpr_values.values()) / max(tpr_values.values())
        if max(tpr_values.values()) > 0
        else 0
    )


def overall_accuracy_equality(y_true, y_pred, sensitive_feature):
    """Overall Accuracy Equality"""
    groups = np.unique(sensitive_feature)
    group_counts = {group: len(y_true[sensitive_feature == group]) for group in groups}

    total_product = np.prod(list(group_counts.values()))
    tt = len(y_true)
    accuracy_values = {
        group: (
            accuracy_score(
                y_true[sensitive_feature == group], y_pred[sensitive_feature == group]
            )
        )
        * (total_product / group_counts[group])
        for group in groups
    }

    return (
        min(accuracy_values.values()) / max(accuracy_values.values())
        if max(accuracy_values.values()) > 0
        else 0
    )


def predictive_equality(y_true, y_pred, sensitive_feature):
    """Predictive Equality (False Positive Rate Parity)"""
    groups = np.unique(sensitive_feature)
    fpr_values = {}
    group_counts = {group: len(y_true[sensitive_feature == group]) for group in groups}

    total_product = np.prod(list(group_counts.values()))
    tt = len(y_true)

    for group in groups:
        tn, fp, fn, tp = confusion_matrix(
            y_true[sensitive_feature == group], y_pred[sensitive_feature == group]
        ).ravel()
        fpr_values[group] = (
            (fp / (fp + tn)) * (total_product / group_counts[group])
            if (fp + tn) > 0
            else 0
        )
    return (
        min(fpr_values.values()) / max(fpr_values.values())
        if max(fpr_values.values()) > 0
        else 0
    )


def predictive_parity(y_true, y_pred, sensitive_feature):
    """Predictive Parity (Positive Predictive Value Parity)"""
    groups = np.unique(sensitive_feature)
    group_counts = {group: len(y_true[sensitive_feature == group]) for group in groups}

    total_product = np.prod(list(group_counts.values()))
    tt = len(y_true)
    tpr_values = {}
    for group in groups:
        tn, fp, fn, tp = confusion_matrix(
            y_true[sensitive_feature == group], y_pred[sensitive_feature == group]
        ).ravel()
        tpr_values[group] = (
            (tp + fp / (fn + tp + fp + tn)) if (fn + tp + fp + tn) > 0 else 0
        )

    return (
        min(tpr_values.values()) / max(tpr_values.values())
        if max(tpr_values.values()) > 0
        else 0
    )


def statistical_parity(y_true, y_pred, sensitive_feature):
    """Statistical Parity"""
    groups = np.unique(sensitive_feature)
    fpr_values = {}
    group_counts = {group: len(y_true[sensitive_feature == group]) for group in groups}

    total_product = np.prod(list(group_counts.values()))

    for group in groups:
        tn, fp, fn, tp = confusion_matrix(
            y_true[sensitive_feature == group], y_pred[sensitive_feature == group]
        ).ravel()
        fpr_values[group] = (
            (fn + tn) * (total_product / group_counts[group]) if (fp + tn) > 0 else 0
        )
    return (
        min(fpr_values.values()) / max(fpr_values.values())
        if max(fpr_values.values()) > 0
        else 0
    )


def treatment_equality(y_true, y_pred, sensitive_feature):
    """Treatment Equality (Ratio of FPR to FNR Parity)"""
    groups = np.unique(sensitive_feature)
    ratio_values = {}
    group_counts = {group: len(y_true[sensitive_feature == group]) for group in groups}

    total_product = np.prod(list(group_counts.values()))
    tt = len(y_true)
    for group in groups:
        tn, fp, fn, tp = confusion_matrix(
            y_true[sensitive_feature == group], y_pred[sensitive_feature == group]
        ).ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        ratio_values[group] = (
            fpr / fnr * (total_product / group_counts[group]) if fnr > 0 else 0
        )
    return (
        min(ratio_values.values()) / max(ratio_values.values())
        if max(ratio_values.values()) > 0
        else 0
    )


def conditional_use_accuracy_equality(y_true, y_pred, sensitive_feature):
    """Conditional Use Accuracy Equality"""
    groups = np.unique(sensitive_feature)
    group_counts = {group: len(y_true[sensitive_feature == group]) for group in groups}

    po = equal_opportunity(y_true, y_pred, sensitive_feature)

    tnr_values = {}
    for group in groups:
        tn, fp, fn, tp = confusion_matrix(
            y_true[sensitive_feature == group], y_pred[sensitive_feature == group]
        ).ravel()
        tnr_values[group] = (tn / (fp + tn)) if (fp + tn) > 0 else 0

    return po - abs((max(tnr_values.values()) - min(tnr_values.values())))


def total_fairness(y_true, y_pred, sensitive_feature):
    """Total Fairness (Combines multiple notions)"""
    notions = [
        statistical_parity(y_true, y_pred, sensitive_feature),
        conditional_statistical_parity(y_true, y_pred, sensitive_feature),
        equal_opportunity(y_true, y_pred, sensitive_feature),
        predictive_equality(y_true, y_pred, sensitive_feature),
    ]
    return np.mean(notions)


def balance_for_positive_class(y_true, y_pred, sensitive_feature):
    groups = np.unique(sensitive_feature)
    balance_values = {}

    for group in groups:
        group_indices = sensitive_feature == group
        y_true_group = y_true[group_indices]
        y_pred_group = y_pred[group_indices]
        tn, fp, fn, tp = confusion_matrix(y_true_group, y_pred_group).ravel()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        balance_values[group] = tpr / fpr

    return (
        min(balance_values.values()) / max(balance_values.values())
        if max(balance_values.values()) > 0
        else 0
    )


def equalized_odds(y_true, y_pred, sensitive_feature):
    """Equalized Odds (Equal TPR and FPR)"""
    groups = np.unique(sensitive_feature)
    fpr_values = {}
    group_counts = {group: len(y_true[sensitive_feature == group]) for group in groups}

    total_product = np.prod(list(group_counts.values()))
    ne = {}
    for group in groups:
        tn, fp, fn, tp = confusion_matrix(
            y_true[sensitive_feature == group], y_pred[sensitive_feature == group]
        ).ravel()
        ne[group] = (tn) / (tn + fp) if (tn + fp) > 0 else 0
    return equal_opportunity(y_true, y_pred, sensitive_feature) - abs(
        (max(ne.values()) - min(ne.values()))
    )


def balance_for_negative_class(y_true, y_pred, sensitive_feature):
    """Balance for Negative Class (TNR - FNR Parity)"""
    groups = np.unique(sensitive_feature)
    balance_values = {}
    for group in groups:
        group_indices = sensitive_feature == group
        y_true_group = y_true[group_indices]
        y_pred_group = y_pred[group_indices]
        tn, fp, fn, tp = confusion_matrix(y_true_group, y_pred_group).ravel()
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        balance_values[group] = tnr / fnr

    return (
        min(balance_values.values()) / max(balance_values.values())
        if max(balance_values.values()) > 0
        else 0
    )


def test_fairness(y_true, y_pred, sensitive_feature, probabilities, n_bins=10):
    """Test-Fairness: Ensures equal precision across protected groups for any value of S."""
    probabilities = (
        np.array(probabilities)[:, 1]
        if probabilities.ndim > 1
        else np.array(probabilities)
    )

    bins = np.linspace(0, 1, n_bins + 1)
    digitized = np.digitize(probabilities, bins) - 1

    group_precision = {group: [] for group in np.unique(sensitive_feature)}

    for group in group_precision:
        for i in range(n_bins):
            bin_mask = (digitized == i) & (sensitive_feature == group)
            if np.any(bin_mask):
                true_labels = y_true[bin_mask]
                pred_labels = probabilities[bin_mask] >= bins[i]
                precision = precision_score(true_labels, pred_labels, zero_division=0)
                group_precision[group].append(precision)
            else:
                group_precision[group].append(None)

    max_disparity = 0
    for i in range(n_bins):
        bin_precisions = [
            group_precision[group][i]
            for group in group_precision
            if group_precision[group][i] is not None
        ]
        if bin_precisions:
            max_precision = max(bin_precisions)
            min_precision = min(bin_precisions)
            disparity = max_precision - min_precision
            if disparity > max_disparity:
                max_disparity = disparity

    return max_disparity


def well_calibration(y_true, y_pred, sensitive_feature, probabilities):
    """Well Calibration (Calibration Error Parity)"""
    groups = np.unique(sensitive_feature)
    calibration_errors = {}
    probabilities = np.array(probabilities)

    for group in groups:
        group_mask = sensitive_feature == group
        group_true = y_true[group_mask]
        group_probs = probabilities[group_mask, 1]

        brier_score = brier_score_loss(group_true, group_probs)
        calibration_errors[group] = brier_score

    max_error = max(calibration_errors.values())
    min_error = min(calibration_errors.values())
    return min_error / max_error if max_error > 0 else 0
