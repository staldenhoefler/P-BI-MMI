"""Multi-layer perceptron classifier (PyTorch Lightning).

The third compared model. A small feed-forward net trained with cross-entropy.
Good when there are non-linear patterns and enough data; on a small tabular
dataset it tends to need careful regularisation and rarely beats gradient
boosting, which is exactly the kind of trade-off the report can discuss.
"""

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAUROC,
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
)


class QuizMLPClassifier(pl.LightningModule):
    """Feed-forward binary classifier for pass/fail prediction."""

    def __init__(self, input_dim, config, class_weights=None):
        super().__init__()
        self.save_hyperparameters(ignore=['config', 'class_weights'])
        self.config = config
        self.learning_rate = config['models']['mlp']['learning_rate']

        hidden_dims = config['models']['mlp']['hidden_dims']
        dropout = config['models']['mlp'].get('dropout', 0.0)

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        # Two output logits for binary classification (failed / passed).
        layers.append(nn.Linear(prev_dim, 2))
        self.network = nn.Sequential(*layers)

        # Class-weighted loss to counter the pass/fail imbalance. Weights are
        # passed in (computed from the training labels) and registered as a
        # buffer so they move to the right device with the model.
        if class_weights is not None:
            weight = torch.as_tensor(class_weights, dtype=torch.float32)
            self.register_buffer('class_weights', weight)
        else:
            self.class_weights = None
        self.loss_fn = nn.CrossEntropyLoss(weight=self.class_weights)

        # Metrics, mirroring src.metrics.compute_metrics keys.
        self.train_acc = BinaryAccuracy()
        self.val_acc = BinaryAccuracy()
        self.test_acc = BinaryAccuracy()
        self.test_precision = BinaryPrecision()
        self.test_recall = BinaryRecall()
        self.test_f1 = BinaryF1Score()
        self.test_auroc = BinaryAUROC()

        # Collected during test_step so the same metrics/plots can be logged
        # to wandb as for the sklearn models.
        self.test_targets: list = []
        self.test_preds: list = []
        self.test_probs: list = []

    def forward(self, x):
        return self.network(x)

    def _positive_prob(self, logits):
        return torch.softmax(logits, dim=1)[:, 1]

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log('train_loss', loss, prog_bar=True)
        self.train_acc(self._positive_prob(logits), y)
        self.log('train_acc', self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log('val_loss', loss, prog_bar=True)
        self.val_acc(self._positive_prob(logits), y)
        self.log('val_acc', self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        prob = self._positive_prob(logits)
        preds = logits.argmax(dim=1)

        self.log('test_loss', loss)
        self.log('test_accuracy', self.test_acc(prob, y), on_epoch=True)
        self.log('test_precision', self.test_precision(preds, y), on_epoch=True)
        self.log('test_recall', self.test_recall(preds, y), on_epoch=True)
        self.log('test_f1', self.test_f1(preds, y), on_epoch=True)
        self.log('test_roc_auc', self.test_auroc(prob, y), on_epoch=True)

        self.test_targets.extend(y.cpu().tolist())
        self.test_preds.extend(preds.cpu().tolist())
        self.test_probs.extend(prob.cpu().tolist())
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
