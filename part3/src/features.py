"""Build the model-ready feature matrix from the merged dataframe.

This is the single source of truth for the train/test split, imputation and
scaling. Every model (Logistic Regression, XGBoost and the MLP) consumes the
*same* split produced here, which is what makes the wandb comparison fair.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.data import get_data

# Columns that must never be used as features: identifiers and anything derived
# from the exam itself (would leak the target). Same list used historically in
# dataset.py and feature_selection.ipynb.
LEAKAGE_COLUMNS = [
    'StudentID', 'Semester',
    'MDM', 'Stat', 'Formalise', 'Cost', 'Final_Exam_Score', 'passed',
]


def resolve_feature_names(data_cfg: dict, df: pd.DataFrame) -> list:
    """Decide which columns to use as features, based on the config.

    ``data.feature_set: "all"`` uses every column except the leakage/identifier
    columns; ``"selected"`` uses only ``data.selected_features``. Selecting a
    leakage column is rejected so the config can never reintroduce target leakage.
    """
    feature_set = data_cfg.get('feature_set', 'all')

    if feature_set == 'selected':
        selected = data_cfg.get('selected_features') or []
        if not selected:
            raise ValueError(
                "data.feature_set is 'selected' but data.selected_features is empty."
            )
        leaking = [c for c in selected if c in LEAKAGE_COLUMNS]
        if leaking:
            raise ValueError(f"selected_features must not contain leakage columns: {leaking}")
        missing = [c for c in selected if c not in df.columns]
        if missing:
            raise ValueError(f"selected_features not present in the data: {missing}")
        return list(selected)

    if feature_set != 'all':
        raise ValueError(f"Unknown data.feature_set '{feature_set}' (use 'all' or 'selected').")

    return [c for c in df.columns if c not in LEAKAGE_COLUMNS]


@dataclass
class Dataset:
    """Container for a single, shared train/test split."""
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: list
    scaler: StandardScaler

    @property
    def input_dim(self) -> int:
        return self.X_train.shape[1]


def build_dataset(config: dict, df: pd.DataFrame | None = None) -> Dataset:
    """Create the shared, scaled train/test split used by all models.

    Args:
        config: parsed config.yaml. Uses ``data.data_folder``,
            ``data.target_variable``, ``data.test_size``, ``data.random_state``
            and the feature-set keys ``data.feature_set`` /
            ``data.selected_features``.
        df: optional pre-loaded dataframe (e.g. from a notebook); if ``None`` it
            is loaded via :func:`src.data.get_data`.

    Returns:
        Dataset with scaled numpy arrays and the fitted scaler.
    """
    data_cfg = config['data']
    target_col = data_cfg.get('target_variable', 'passed')

    if df is None:
        df = get_data(data_cfg.get('data_folder', 'data'))

    df = df.fillna(0)

    # Target as int (0/1) so it works for sklearn and torch CrossEntropy alike.
    y = df[target_col].astype(int).values

    feature_names = resolve_feature_names(data_cfg, df)
    X = df[feature_names].values.astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=data_cfg.get('test_size', 0.2),
        random_state=data_cfg.get('random_state', 42),
        stratify=y,
    )

    # Fit scaler ONLY on the training data to avoid leakage.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    return Dataset(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=feature_names,
        scaler=scaler,
    )
