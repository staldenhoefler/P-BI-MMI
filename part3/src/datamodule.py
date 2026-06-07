"""PyTorch Lightning DataModule for the MLP classifier.

Wraps the shared split from :func:`src.features.build_dataset` in tensors and
dataloaders so the MLP trains on exactly the same data as the sklearn models.
"""

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset, random_split

from src.features import build_dataset


class QuizDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        # Class indices (long) for nn.CrossEntropyLoss.
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class QuizDataModule(pl.LightningDataModule):
    def __init__(self, config, dataset=None):
        super().__init__()
        self.config = config
        self.batch_size = config['data']['batch_size']
        self.val_size = config['data'].get('val_size', 0.1)
        self.random_state = config['data'].get('random_state', 42)
        # Allow passing a pre-built Dataset so the MLP shares the exact same
        # split as the sklearn models; otherwise build it here.
        self._dataset = dataset
        self.input_dim = None

    def setup(self, stage=None):
        ds = self._dataset or build_dataset(self.config)
        self.input_dim = ds.input_dim

        full_train = QuizDataset(ds.X_train, ds.y_train)
        # Carve a validation set out of the training split (relative to whole set).
        val_prop = self.val_size / (1.0 - self.config['data'].get('test_size', 0.2))
        n_val = max(1, int(len(full_train) * val_prop))
        n_train = len(full_train) - n_val
        generator = torch.Generator().manual_seed(self.random_state)
        self.train_dataset, self.val_dataset = random_split(
            full_train, [n_train, n_val], generator=generator
        )
        self.test_dataset = QuizDataset(ds.X_test, ds.y_test)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)
