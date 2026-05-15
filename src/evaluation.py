import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def compute_metrics(y_true, y_pred, y_prob=None):
    """
    Compute standard binary classification metrics.
    Positive class is assumed to be 1.
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    sensitivity = recall_score(y_true, y_pred, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    auc = (
        roc_auc_score(y_true, y_prob)
        if y_prob is not None and len(np.unique(y_true)) == 2
        else np.nan
    )

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "AUC-ROC": auc,
    }


def threshold_by_positive_quantile(y_true, y_prob, target_sensitivity=0.90):
    """
    Select threshold based on positive-class score quantile.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    positive_scores = y_prob[y_true == 1]

    if len(positive_scores) == 0:
        return 0.5

    quantile = max(0.0, min(1.0, 1.0 - target_sensitivity))
    return float(np.quantile(positive_scores, quantile))


def make_results_row(model_name, metrics):
    """
    Convert a metrics dictionary into a one-row DataFrame.
    """
    row = {"Model": model_name}
    row.update(metrics)

    return pd.DataFrame([row])


def plot_roc_curves(curves, title="ROC Curve", save_path=None):
    """
    Plot ROC curves for multiple models.

    curves should be a list of:
    (model_name, y_true, y_score)
    """
    plt.figure(figsize=(7, 6))

    for model_name, y_true, y_score in curves:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc_value = roc_auc_score(y_true, y_score)

        plt.plot(fpr, tpr, label=f"{model_name} (AUC = {auc_value:.3f})")

    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Sensitivity)")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

    if save_path:
        plt.savefig(save_path, dpi=160, bbox_inches="tight")

    return plt.gca()
