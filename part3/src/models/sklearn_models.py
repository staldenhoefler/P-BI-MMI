"""Scikit-learn / XGBoost classifier factories.

Two of the three compared models live here. They are deliberately simple and
interpretable so the report can argue *why* each is a good or bad fit:

- Logistic Regression: linear, fast, highly interpretable baseline. Good when
  the pass/fail boundary is roughly linear in the quiz features; weak if the
  relationship is non-linear or features interact.
- XGBoost: gradient-boosted trees, the strong default for small/medium tabular
  data. Captures non-linearities and interactions and tolerates unscaled,
  correlated features; risk of overfitting on a small dataset.
"""

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


def build_logreg(config: dict, y_train=None) -> LogisticRegression:
    """Build a Logistic Regression classifier from ``config['models']['logreg']``.

    ``y_train`` is accepted for a uniform factory signature but unused here.
    """
    params = config.get('models', {}).get('logreg', {})
    return LogisticRegression(
        max_iter=params.get('max_iter', 10000),
        penalty=params.get('penalty', 'l2'),
        solver=params.get('solver', 'lbfgs'),
        C=params.get('C', 1.0),
        class_weight=params.get('class_weight', None),
        random_state=config['data'].get('random_state', 42),
    )


def build_xgboost(config: dict, y_train=None) -> XGBClassifier:
    """Build an XGBoost classifier from ``config['models']['xgboost']``.

    If ``y_train`` is given, ``scale_pos_weight`` is set to
    ``n_negative / n_positive`` so the (minority) positive class is up-weighted,
    which matters on this small, imbalanced dataset.
    """
    params = config.get('models', {}).get('xgboost', {})

    scale_pos_weight = params.get('scale_pos_weight', 1.0)
    if y_train is not None:
        y_train = np.asarray(y_train)
        n_pos = int((y_train == 1).sum())
        n_neg = int((y_train == 0).sum())
        if n_pos > 0:
            scale_pos_weight = n_neg / n_pos

    return XGBClassifier(
        n_estimators=params.get('n_estimators', 100),
        max_depth=params.get('max_depth', 3),
        learning_rate=params.get('learning_rate', 0.1),
        subsample=params.get('subsample', 1.0),
        eval_metric=params.get('eval_metric', 'logloss'),
        scale_pos_weight=scale_pos_weight,
        random_state=config['data'].get('random_state', 42),
    )


def build_dummy(config: dict, y_train=None, strategy: str = 'most_frequent') -> DummyClassifier:
    """Build a baseline :class:`DummyClassifier` (no features used).

    ``strategy='most_frequent'`` always predicts the majority class; ``'stratified'``
    samples predictions from the training class distribution. These give the floor
    every real model must beat, routed through the same metric/logging path.
    """
    return DummyClassifier(
        strategy=strategy,
        random_state=config['data'].get('random_state', 42),
    )
