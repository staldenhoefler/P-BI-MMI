"""Shared metric computation and Weights & Biases logging.

Both the sklearn models and the Lightning MLP route their test predictions
through here, so every model ends up with the *same* metric keys and the same
plots in wandb. This is what makes "log every model" a real comparison rather
than apples to oranges.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

import wandb

# Human-readable class names for confusion-matrix / ROC plots.
CLASS_NAMES = ["failed", "passed"]


def compute_metrics(y_true, y_pred, y_prob=None) -> dict:
    """Compute the standard binary-classification metrics as a flat dict.

    Args:
        y_true: ground-truth labels (0/1).
        y_pred: predicted labels (0/1).
        y_prob: optional probability of the positive class, needed for ROC-AUC.
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_prob is not None and len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
    return metrics


def log_run_to_wandb(name, config, y_true, y_pred, y_prob, project="quiz_prediction"):
    """Log a single model's results as its own wandb run.

    Mirrors what Lightning's WandbLogger does for the MLP: a dedicated run in the
    shared project, scalar metrics, plus a confusion matrix and ROC curve.

    Returns the computed metrics dict (also useful for printing / notebooks).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob) if y_prob is not None else None

    metrics = compute_metrics(y_true, y_pred, y_prob)

    run = wandb.init(project=project, name=name, config=config, reinit=True)
    run.log(metrics)
    run.log({
        "confusion_matrix": wandb.plot.confusion_matrix(
            y_true=y_true.tolist(),
            preds=y_pred.tolist(),
            class_names=CLASS_NAMES,
        )
    })
    if y_prob is not None:
        # wandb.plot.roc_curve expects per-class probabilities, shape (n, n_classes).
        probs_2col = np.column_stack([1 - y_prob, y_prob])
        run.log({
            "roc_curve": wandb.plot.roc_curve(
                y_true=y_true,
                y_probas=probs_2col,
                labels=CLASS_NAMES,
            )
        })
    run.summary.update(metrics)
    run.finish()

    return metrics
