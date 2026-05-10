import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
from utility import get_data

class QuizDataset(Dataset):
    def __init__(self, X, y):
        """
        Args:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): Target vector.
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1) # [batch_size, 1]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class QuizDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.data_folder = self.config['data'].get('data_folder', 'data')
        self.target_col = self.config['data']['target_variable']
        self.batch_size = self.config['data']['batch_size']
        self.test_size = self.config['data']['test_size']
        self.val_size = self.config['data']['val_size']
        self.random_state = self.config['data']['random_state']
        
        self.scaler = StandardScaler()

    def prepare_data(self):
        # Optional: any one-time downloads or data processing
        pass

    def setup(self, stage=None):
        # 1. Load the merged dataset
        df = get_data(self.data_folder)

        df = df.fillna(0)

        # 2. Define the target and separate out features
        exam_columns = ['MDM', 'Stat', 'Formalise', 'Cost', 'Final_Exam_Score']
            
        y = df[self.target_col].values
        
        # We drop the identifiers and ALL exam columns to prevent data leakage
        cols_to_drop = ['StudentID', 'Semester'] + exam_columns + [self.target_col]
        
        # Select only the features that are not in the drop list
        feature_cols = [col for col in df.columns if col not in cols_to_drop]
        X = df[feature_cols].values
        
        # 3. Train / Val / Test split
        # First split into Train and Test
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        
        # Then split Train into Train and Validation
        # The val_size in config is usually relative to the whole dataset, 
        # so we adjust the proportion here:
        val_prop = self.val_size / (1.0 - self.test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_prop, random_state=self.random_state
        )
        
        # 4. Standardize features
        # Fit scaler ONLY on training data to prevent data leakage
        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)
        X_test = self.scaler.transform(X_test)
        
        # 5. Store input dimension for the model
        self.input_dim = X_train.shape[1]
        
        # 6. Create Datasets
        if stage == 'fit' or stage is None:
            self.train_dataset = QuizDataset(X_train, y_train)
            self.val_dataset = QuizDataset(X_val, y_val)
            
        if stage == 'test' or stage is None:
            self.test_dataset = QuizDataset(X_test, y_test)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)
