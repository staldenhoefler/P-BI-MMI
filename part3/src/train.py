"""Train and log all three models to Weights & Biases.

Running ``python -m src.train`` (from the ``part3/`` folder) trains Logistic
Regression, XGBoost and the MLP on the *same* train/test split and produces
three runs in the wandb project, each with the same metric keys (accuracy,
precision, recall, f1, roc_auc) plus a confusion matrix and ROC curve.
"""

import argparse

import numpy as np
import pytorch_lightning as pl
import yaml
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from sklearn.utils.class_weight import compute_class_weight

from src.datamodule import QuizDataModule
from src.features import build_dataset
from src.metrics import compute_metrics, log_run_to_wandb
from src.models.mlp import QuizMLPClassifier
from src.models.sklearn_models import build_dummy, build_logreg, build_xgboost


def load_config(path='config.yaml'):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def train_sklearn_model(name, build_fn, dataset, config):
    """Fit a sklearn-style classifier on the shared split and log it to wandb.

    ``build_fn`` is called with ``(config, y_train)`` so models that need the
    training labels (e.g. XGBoost's ``scale_pos_weight``) can use them; factories
    that don't simply ignore the second argument.
    """
    print(f"\n=== Training {name} ===")
    model = build_fn(config, dataset.y_train)
    model.fit(dataset.X_train, dataset.y_train)

    y_pred = model.predict(dataset.X_test)
    y_prob = model.predict_proba(dataset.X_test)[:, 1]

    metrics = log_run_to_wandb(
        name=name,
        config=config,
        y_true=dataset.y_test,
        y_pred=y_pred,
        y_prob=y_prob,
        project=config['logging']['project'],
    )
    print(f"{name}: " + ", ".join(f"{k}={v:.3f}" for k, v in metrics.items()))
    return metrics


def _mlp_class_weights(config, y_train):
    """Balanced class weights for the MLP loss, or None if not requested."""
    if config['models']['mlp'].get('class_weight') != 'balanced':
        return None
    classes = np.array([0, 1])
    return compute_class_weight('balanced', classes=classes, y=np.asarray(y_train))


def train_mlp(name, dataset, config):
    """Train the Lightning MLP on the shared split and log it to wandb."""
    print(f"\n=== Training {name} ===")
    datamodule = QuizDataModule(config, dataset=dataset)
    datamodule.setup('fit')

    class_weights = _mlp_class_weights(config, dataset.y_train)
    model = QuizMLPClassifier(
        input_dim=datamodule.input_dim, config=config, class_weights=class_weights,
    )

    wandb_logger = WandbLogger(
        project=config['logging']['project'],
        name=name,
        config=config,
    )
    callbacks = []
    mlp_cfg = config['models']['mlp']
    if mlp_cfg.get('early_stopping', False):
        callbacks.append(EarlyStopping(
            monitor='val_loss', patience=mlp_cfg.get('patience', 5), mode='min',
        ))
    trainer = pl.Trainer(
        max_epochs=config['trainer']['max_epochs'],
        logger=wandb_logger,
        enable_checkpointing=False,
        log_every_n_steps=1,
        callbacks=callbacks,
    )
    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)

    # Recompute with the shared metric helper so keys match the other models,
    # then log the confusion matrix / ROC curve onto the same run.
    metrics = compute_metrics(
        np.array(model.test_targets),
        np.array(model.test_preds),
        np.array(model.test_probs),
    )
    import wandb
    wandb.log({
        "confusion_matrix": wandb.plot.confusion_matrix(
            y_true=model.test_targets, preds=model.test_preds,
            class_names=["failed", "passed"],
        )
    })
    wandb_logger.experiment.summary.update(metrics)
    wandb.finish()

    print("MLP: " + ", ".join(f"{k}={v:.3f}" for k, v in metrics.items()))
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train all models and log to wandb.")
    parser.add_argument('--config', default='config.yaml')
    args = parser.parse_args()

    config = load_config(args.config)
    print(f"Target variable: {config['data']['target_variable']}")
    print(f"Feature set: {config['data'].get('feature_set', 'all')}")

    # One shared, scaled split for every model -> fair comparison.
    dataset = build_dataset(config)
    print(f"Features: {dataset.input_dim} | train: {len(dataset.y_train)} | test: {len(dataset.y_test)}")

    # Run names are configurable per model in config.yaml -> logging.run_names.
    run_names = config['logging'].get('run_names', {})

    results = {}

    # Naive baselines first: every real model has to beat these. They use no
    # features and go through the exact same compute_metrics / logging path.
    results['Dummy (most_frequent)'] = train_sklearn_model(
        run_names.get('dummy_most_frequent', 'Dummy_most_frequent'),
        lambda cfg, y: build_dummy(cfg, y, strategy='most_frequent'), dataset, config)
    results['Dummy (stratified)'] = train_sklearn_model(
        run_names.get('dummy_stratified', 'Dummy_stratified'),
        lambda cfg, y: build_dummy(cfg, y, strategy='stratified'), dataset, config)

    results['LogisticRegression'] = train_sklearn_model(
        run_names.get('logreg', 'LogisticRegression'), build_logreg, dataset, config)
    results['XGBoost'] = train_sklearn_model(
        run_names.get('xgboost', 'XGBoost'), build_xgboost, dataset, config)
    results['MLP'] = train_mlp(run_names.get('mlp', 'MLP'), dataset, config)

    print("\n=== Summary ===")
    for name, metrics in results.items():
        print(f"{name}: " + ", ".join(f"{k}={v:.3f}" for k, v in metrics.items()))


if __name__ == '__main__':
    main()
