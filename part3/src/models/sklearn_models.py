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

from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


def build_logreg(config: dict) -> LogisticRegression:
    """Build a Logistic Regression classifier from ``config['models']['logreg']``."""
    params = config.get('models', {}).get('logreg', {})
    return LogisticRegression(
        max_iter=params.get('max_iter', 10000),
        C=params.get('C', 1.0),
        class_weight=params.get('class_weight', None),
        random_state=config['data'].get('random_state', 42),
    )


def build_xgboost(config: dict) -> XGBClassifier:
    """Build an XGBoost classifier from ``config['models']['xgboost']``."""
    params = config.get('models', {}).get('xgboost', {})
    return XGBClassifier(
        n_estimators=params.get('n_estimators', 100),
        max_depth=params.get('max_depth', 3),
        learning_rate=params.get('learning_rate', 0.1),
        subsample=params.get('subsample', 1.0),
        eval_metric=params.get('eval_metric', 'logloss'),
        random_state=config['data'].get('random_state', 42),
    )
