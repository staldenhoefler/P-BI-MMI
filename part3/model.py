import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from torchmetrics.regression import R2Score

class QuizModel(pl.LightningModule):
    def __init__(self, input_dim, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.learning_rate = self.config['model']['learning_rate']
        
        # Build network dynamically based on hidden dims
        hidden_dims = self.config['model']['hidden_dims']
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
            
        # Output layer for regression (1 continuous value)
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
        # Loss function for regression
        self.loss_fn = nn.MSELoss()
        
        # Metrics
        self.train_r2 = R2Score()
        self.val_r2 = R2Score()
        self.test_r2 = R2Score()

    def forward(self, x):
        return self.network(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log('train_loss', loss, prog_bar=True)
        
        self.train_r2(y_hat, y)
        self.log('train_r2', self.train_r2, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log('val_loss', loss, prog_bar=True)
        
        self.val_r2(y_hat, y)
        self.log('val_r2', self.val_r2, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log('test_loss', loss)
        
        self.test_r2(y_hat, y)
        self.log('test_r2', self.test_r2, on_step=False, on_epoch=True)
        
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
