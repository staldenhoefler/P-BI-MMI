"""Shared metric computation and Weights & Biases logging.

Both the sklearn models and the Lightning MLP route their test predictions
through here, so every model ends up with the *same* metric keys and the same
plots in wandb. This is what makes "log every model" a real comparison rather
than apples to oranges.
"""

import numpy as np
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

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


# --------------------------------------------------------------------------- #
# Stratified cross-validation (robust on a tiny, imbalanced dataset)
# --------------------------------------------------------------------------- #

def evaluate_cv(estimator, X, y, n_splits=5, random_state=42, scale=True,
                cost_fp=None, cost_fn=None):
    """Stratified-K-fold evaluation on *pooled* out-of-fold predictions.

    A single train/test split on ~135 students is far too noisy to trust. Instead
    we run ``StratifiedKFold(shuffle=True)`` and collect every sample's
    prediction from the fold in which it was held out, then build one confusion
    matrix and one ROC-AUC over all of them. We also report per-fold AUC (and,
    if costs are given, per-fold expected cost) so the notebook can show
    ``mean ± std`` across folds.

    Args:
        estimator: an *unfitted* sklearn-style classifier with ``predict_proba``.
        X, y: the full (unsplit) feature matrix and labels.
        n_splits: number of stratified folds.
        random_state: seed for the shuffle (kept at 42 for reproducibility).
        scale: if True, wrap the estimator in a per-fold ``StandardScaler`` so no
            test-fold statistics leak into training.
        cost_fp, cost_fn: if both given, also report per-fold and pooled expected
            cost at the 0.5 threshold via :func:`expected_cost`.

    Returns:
        dict with ``pooled_metrics`` (accuracy/precision/recall/f1/roc_auc on the
        pooled out-of-fold predictions), ``confusion_matrix`` (2x2 ndarray),
        ``oof_prob`` / ``oof_pred`` (per-sample), ``fold_aucs`` and the summary
        ``auc_mean`` / ``auc_std``. When costs are given it also includes
        ``pooled_cost``, ``fold_costs``, ``cost_mean`` and ``cost_std``.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    pipe = make_pipeline(StandardScaler(), estimator) if scale else estimator
    with_cost = cost_fp is not None and cost_fn is not None

    # Pooled out-of-fold probabilities for the positive class.
    oof_prob = cross_val_predict(pipe, X, y, cv=cv, method='predict_proba')[:, 1]
    oof_pred = (oof_prob >= 0.5).astype(int)

    pooled_metrics = compute_metrics(y, oof_pred, oof_prob)
    cm = confusion_matrix(y, oof_pred, labels=[0, 1])

    # Per-fold AUC (and cost) -> mean ± std (a fold with only one class is skipped).
    fold_aucs = []
    fold_costs = []
    for train_idx, test_idx in cv.split(X, y):
        fold_est = clone(pipe)
        fold_est.fit(X[train_idx], y[train_idx])
        prob = fold_est.predict_proba(X[test_idx])[:, 1]
        if len(np.unique(y[test_idx])) > 1:
            fold_aucs.append(roc_auc_score(y[test_idx], prob))
        if with_cost:
            pred = (prob >= 0.5).astype(int)
            tn, fp, fn, tp = confusion_matrix(y[test_idx], pred, labels=[0, 1]).ravel()
            fold_costs.append(expected_cost(tn, fp, fn, tp, cost_fp, cost_fn))

    result = {
        "pooled_metrics": pooled_metrics,
        "confusion_matrix": cm,
        "oof_prob": oof_prob,
        "oof_pred": oof_pred,
        "fold_aucs": fold_aucs,
        "auc_mean": float(np.mean(fold_aucs)) if fold_aucs else float("nan"),
        "auc_std": float(np.std(fold_aucs)) if fold_aucs else float("nan"),
    }
    if with_cost:
        tn, fp, fn, tp = cm.ravel()
        result["pooled_cost"] = expected_cost(tn, fp, fn, tp, cost_fp, cost_fn)
        result["fold_costs"] = fold_costs
        result["cost_mean"] = float(np.mean(fold_costs)) if fold_costs else float("nan")
        result["cost_std"] = float(np.std(fold_costs)) if fold_costs else float("nan")
    return result


# --------------------------------------------------------------------------- #
# Cost-sensitive evaluation (a false negative hurts more than a false positive)
# --------------------------------------------------------------------------- #

def expected_cost(tn, fp, fn, tp, cost_fp=1.0, cost_fn=5.0, cost_tp=0.0, cost_tn=0.0):
    """Expected (per-sample) misclassification cost from confusion-matrix counts.

    Accuracy treats both error types as equal; here a false negative can be made
    far more expensive than a false positive (``cost_fn >> cost_fp``), which is
    the realistic framing for an early-warning model. Returns total cost divided
    by the number of samples so models/thresholds are comparable.
    """
    total = tn + fp + fn + tp
    if total == 0:
        return 0.0
    cost = fp * cost_fp + fn * cost_fn + tp * cost_tp + tn * cost_tn
    return cost / total


def sweep_cost_threshold(y_true, y_prob, cost_fp=1.0, cost_fn=5.0, thresholds=None):
    """Sweep the decision threshold to find the cost-minimising operating point.

    The default 0.5 cutoff minimises error count, not cost. By varying the
    threshold on the positive-class probability and scoring each with
    :func:`expected_cost`, we find the threshold that minimises expected cost
    given the FN/FP cost ratio.

    Returns:
        dict with ``best_threshold``, ``best_cost``, the cost at the default 0.5
        threshold (``cost_at_0.5``), and ``sweep`` (list of ``(threshold, cost)``).
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 101)

    sweep = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        sweep.append((float(t), expected_cost(tn, fp, fn, tp, cost_fp, cost_fn)))

    best_threshold, best_cost = min(sweep, key=lambda r: r[1])
    tn, fp, fn, tp = confusion_matrix(
        y_true, (y_prob >= 0.5).astype(int), labels=[0, 1]
    ).ravel()
    return {
        "best_threshold": best_threshold,
        "best_cost": best_cost,
        "cost_at_0.5": expected_cost(tn, fp, fn, tp, cost_fp, cost_fn),
        "sweep": sweep,
    }
